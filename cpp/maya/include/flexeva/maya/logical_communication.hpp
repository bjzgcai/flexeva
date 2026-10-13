#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flexeva::maya {

// Explicit fixed-group communication for a captured C++/LibTorch window.
// Values use NCCL datatype/reduction integers. The same exported FakeNCCL ABI
// is used by the Python adapter. No Torch or CUDA headers enter this contract.
class LogicalCommunicator {
public:
    using Registration = std::function<void(std::uint64_t, std::span<const std::uint32_t>, std::uint32_t)>;
    LogicalCommunicator(std::string_view identity, std::span<const std::uint32_t> members,
                        std::uint32_t rank, Registration registration)
        : members_(members.begin(), members.end()) {
        if (identity.empty() || identity.size() > 96 || members_.empty() ||
            members_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || !registration)
            throw std::invalid_argument("invalid fixed communicator registration");
        auto sorted = members_;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument("duplicate communicator member");
        const auto found = std::find(members_.begin(), members_.end(), rank);
        if (found == members_.end()) throw std::invalid_argument("rank outside registered communicator");
        local_rank_ = static_cast<int>(found - members_.begin());
        library_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw std::runtime_error("cannot load FakeNCCL");
        try {
            if (!symbol<int(*)()>("fakecuda_ras_active")())
                throw std::runtime_error("logical communicator requires an active direct capture");
            // Identical registered group names produce identical UID bytes at
            // each sequential rank. Membership is recorded by the callback.
            UniqueId uid{};
            constexpr std::string_view prefix = "flexeva-fixed:";
            std::memcpy(uid.bytes.data(), prefix.data(), prefix.size());
            std::memcpy(uid.bytes.data() + prefix.size(), identity.data(), identity.size());
            check(symbol<int(*)(void**, int, UniqueId, int)>("ncclCommInitRank")(
                &handle_, static_cast<int>(members_.size()), uid, local_rank_));
            registration(reinterpret_cast<std::uintptr_t>(handle_), members_, local_rank_);
        } catch (...) { close(); throw; }
    }
    LogicalCommunicator(const LogicalCommunicator&) = delete;
    LogicalCommunicator& operator=(const LogicalCommunicator&) = delete;
    ~LogicalCommunicator() { close(); }

    void all_reduce(const void* input, void* output, std::size_t count, int datatype,
                    int reduction = 0, void* stream = nullptr) {
        check(symbol<int(*)(const void*,void*,std::size_t,int,int,void*,void*)>("ncclAllReduce")(
            input,output,count,datatype,reduction,handle_,stream));
    }
    void broadcast(const void* input, void* output, std::size_t count, int datatype,
                   std::uint32_t root, void* stream = nullptr) {
        check(symbol<int(*)(const void*,void*,std::size_t,int,int,void*,void*)>("ncclBroadcast")(
            input,output,count,datatype,local(root),handle_,stream));
    }
    void all_gather(const void* input, void* output, std::size_t send_count, int datatype,
                    void* stream = nullptr) {
        check(symbol<int(*)(const void*,void*,std::size_t,int,void*,void*)>("ncclAllGather")(
            input,output,send_count,datatype,handle_,stream));
    }
    void reduce_scatter(const void* input, void* output, std::size_t receive_count, int datatype,
                        int reduction = 0, void* stream = nullptr) {
        check(symbol<int(*)(const void*,void*,std::size_t,int,int,void*,void*)>("ncclReduceScatter")(
            input,output,receive_count,datatype,reduction,handle_,stream));
    }
    void send(const void* input, std::size_t count, int datatype, std::uint32_t peer, void* stream = nullptr) {
        check(symbol<int(*)(const void*,std::size_t,int,int,void*,void*)>("ncclSend")(
            input,count,datatype,local(peer),handle_,stream));
    }
    void receive(void* output, std::size_t count, int datatype, std::uint32_t peer, void* stream = nullptr) {
        check(symbol<int(*)(void*,std::size_t,int,int,void*,void*)>("ncclRecv")(
            output,count,datatype,local(peer),handle_,stream));
    }
    void all_to_all(const void* input, void* output, std::span<const std::size_t> send_counts,
                    std::span<const std::size_t> receive_counts, int datatype, void* stream = nullptr) {
        if (send_counts.size() != members_.size() || receive_counts.size() != members_.size())
            throw std::invalid_argument("all-to-all requires one count per fixed group member");
        const auto width = datatype_width(datatype);
        const auto total = [&](auto counts) {
            std::size_t sum = 0;
            for (auto count : counts) {
                if (count > std::numeric_limits<std::size_t>::max() / width - sum)
                    throw std::overflow_error("all-to-all byte count overflow");
                sum += count;
            }
            return sum * width;
        };
        const auto send_bytes = total(send_counts), receive_bytes = total(receive_counts);
        if ((send_bytes && !input) || (receive_bytes && !output))
            throw std::invalid_argument("null all-to-all buffer");
        check(symbol<int(*)()>("ncclGroupStart")());
        try {
            std::size_t send_offset = 0, receive_offset = 0;
            for (std::size_t peer = 0; peer < members_.size(); ++peer) {
                if (send_counts[peer]) send(static_cast<const char*>(input) + send_offset,
                    send_counts[peer], datatype, members_[peer], stream);
                if (receive_counts[peer]) receive(static_cast<char*>(output) + receive_offset,
                    receive_counts[peer], datatype, members_[peer], stream);
                send_offset += send_counts[peer] * width;
                receive_offset += receive_counts[peer] * width;
            }
        } catch (...) { symbol<int(*)()>("ncclGroupEnd")(); throw; }
        check(symbol<int(*)()>("ncclGroupEnd")());
    }
    // A synchronization collective has no numerical payload. CUDA stream
    // synchronization by the caller supplies the desired host wait relation.
    void barrier(void* stream = nullptr) { all_reduce(nullptr,nullptr,0,7,0,stream); }
    void finish() {
        if (handle_) {
            auto* handle = handle_; handle_ = nullptr;
            check(symbol<int(*)(void*)>("ncclCommDestroy")(handle));
        }
    }
private:
    struct UniqueId { std::array<unsigned char,128> bytes; };
    void* library_ = nullptr;
    void* handle_ = nullptr;
    std::vector<std::uint32_t> members_;
    int local_rank_ = 0;
    template<class Function> Function symbol(const char* name) const {
        auto* address = dlsym(library_,name);
        if (!address) throw std::runtime_error(std::string("missing FakeNCCL symbol: ") + name);
        return reinterpret_cast<Function>(address);
    }
    static void check(int result) {
        if (result) throw std::runtime_error("logical NCCL operation failed: " + std::to_string(result));
    }
    int local(std::uint32_t rank) const {
        const auto found = std::find(members_.begin(),members_.end(),rank);
        if (found == members_.end()) throw std::invalid_argument("peer outside fixed communicator");
        return static_cast<int>(found-members_.begin());
    }
    static std::size_t datatype_width(int type) {
        switch(type) {
        case 0:case 1:return 1;
        case 6:case 9:return 2;
        case 2:case 3:case 7:return 4;
        case 4:case 5:case 8:return 8;
        default:throw std::invalid_argument("unsupported NCCL datatype");
        }
    }
    void close() noexcept {
        if (handle_) {
            auto destroy = reinterpret_cast<int(*)(void*)>(dlsym(library_,"ncclCommDestroy"));
            if (destroy) destroy(handle_);
            handle_ = nullptr;
        }
        if (library_) { dlclose(library_); library_ = nullptr; }
    }
};
} // namespace flexeva::maya
