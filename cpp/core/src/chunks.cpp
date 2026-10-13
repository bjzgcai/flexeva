#include "runtime_internal.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string_view>
#include <utility>

namespace flexeva::core::runtime {

ChunkOutput::ChunkOutput(std::shared_ptr<const detail::ChunkOutputData> value) : data_(std::move(value)) {}
ChunkOutput::operator bool() const noexcept { return bool(data_); }
std::size_t ChunkOutput::count() const noexcept { return data_ ? data_->count : 0; }
std::size_t ChunkOutput::size() const noexcept { return data_ ? data_->bytes : 0; }
BlockBundle ChunkOutput::storage() const { return data_ ? data_->storage : BlockBundle{}; }
bool ChunkOutput::same_storage(const ChunkOutput& other) const noexcept {
    if (!data_ || !other.data_) return !data_ && !other.data_;
    const auto* a=detail::BundleAccess::get(data_->storage);
    const auto* b=detail::BundleAccess::get(other.data_->storage);
    return a && b && a->directory==b->directory && a->id==b->id;
}

namespace detail {
std::vector<BlockBundle> receipt_bundles(const Receipt& receipt) {
    auto result=receipt.bundles;
    if(receipt.chunks) result.push_back(receipt.chunks.storage());
    return result;
}
} // namespace detail

struct ChunkWriter::Impl {
    std::shared_ptr<detail::Access> access;
    std::shared_ptr<detail::ChunkOutputData> output;
    std::shared_ptr<detail::BlockBundleData> storage;
    ras::BufferWriter writer;
    std::size_t block_bytes;
    ras::Layout layout;
    bool active = false;
    bool intern;

    Impl(std::shared_ptr<detail::Access> a, std::size_t bytes, ras::Layout value, bool share)
        : access(std::move(a)), output(std::make_shared<detail::ChunkOutputData>()), block_bytes(bytes), layout(value), intern(share) {
        detail::check(*access);
        if (access->phase != detail::Phase::Apply || !access->isolated || !access->transform || !access->scope->tx)
            throw ras::Error("chunk output requires an isolated Apply Boundary");
        if (!block_bytes || !layout.element_size || !std::has_single_bit(layout.alignment) ||
            layout.element_size % layout.alignment != 0)
            throw ras::Error("invalid Core chunk block size or layout");
        const auto remainder = block_bytes % layout.element_size;
        if (remainder) {
            const auto padding = layout.element_size - remainder;
            if (block_bytes > std::numeric_limits<std::size_t>::max() - padding)
                throw ras::Error("Core chunk block size overflow");
            block_bytes += padding;
        }
        storage=std::make_shared<detail::BlockBundleData>();
        storage->directory=access->scope->tx->add_directory();
        detail::track_bundle_directory(*access,storage->directory);
        storage->id=access->scope->tx->new_change_id().value;
        storage->layout=layout;
        output->boundary = access->isolated->id;
        output->layout = layout;
        if (!access->chunk_outputs->emplace(output->boundary, output).second)
            throw ras::Error("Boundary already opened a chunk output");
    }

    ras::Buffer intern_block(ras::Buffer buffer) {
        auto& scope = *access->scope;
        if (!scope.chunk_intern) scope.chunk_intern = std::make_unique<detail::ChunkInternPool>();
        const auto view = buffer.bytes();
        const auto bytes = view.span();
        const auto hash = std::hash<std::string_view>{}({reinterpret_cast<const char*>(bytes.data()), bytes.size()});
        if (scope.counters) scope.counters->chunk_intern_hash_bytes += bytes.size();
        auto& bucket = scope.chunk_intern->buckets[hash];
        for (const auto& candidate : bucket) {
            if (candidate.size() != bytes.size() || candidate.layout() != buffer.layout()) continue;
            const auto existing = candidate.bytes();
            if (scope.counters) scope.counters->chunk_intern_compare_bytes += bytes.size();
            // A hash match is only an index lookup; complete byte equality is
            // required before replacing this Boundary's own immutable output.
            if (std::memcmp(existing.span().data(), bytes.data(), bytes.size()) == 0) {
                if (scope.counters) {
                    scope.counters->chunk_intern_shared_bytes += bytes.size();
                    ++scope.counters->chunk_intern_reused_blocks;
                }
                return candidate;
            }
        }
        bucket.push_back(buffer);
        return buffer;
    }

    void check() const {
        detail::check(*access);
        if (output->finished) throw ras::Error("Core chunk writer is already finished");
    }

    void seal() {
        if (!active) return;
        const auto used = writer.size();
        // finish_prefix also checks outstanding/abandoned borrows, including
        // an empty committed prefix. No bytes outside that prefix become public.
        auto buffer = writer.finish_prefix();
        active = false;
        if (!used) return;
        if (output->bytes > std::numeric_limits<std::size_t>::max() - used ||
            output->count == std::numeric_limits<std::size_t>::max())
            throw ras::Error("Core chunk output size overflow");
        if (intern) buffer = intern_block(std::move(buffer));
        access->scope->tx->directory_put(storage->directory, {storage->id, output->count}, {{}, buffer, {}});
        detail::count_block_write(access->scope->counters.get());
        ++output->count;
        output->bytes += used;
    }
};

ChunkWriter::ChunkWriter(std::shared_ptr<detail::Access> access, std::size_t bytes, ras::Layout layout, bool intern)
    : impl_(std::make_unique<Impl>(std::move(access), bytes, layout, intern)) {}
ChunkWriter::ChunkWriter(ChunkWriter&&) noexcept = default;
ChunkWriter& ChunkWriter::operator=(ChunkWriter&&) noexcept = default;
ChunkWriter::~ChunkWriter() = default;

ras::WriteSpan ChunkWriter::reserve(std::size_t bytes) {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    if (!bytes || bytes % impl_->layout.element_size != 0)
        throw ras::Error("Core chunk reservation splits an element or is empty");
    if (impl_->active && bytes > impl_->writer.capacity() - impl_->writer.size()) impl_->seal();
    if (!impl_->active) {
        impl_->writer = impl_->access->scope->tx->buffer(std::max(bytes, impl_->block_bytes), impl_->layout);
        impl_->active = true;
    }
    return impl_->writer.reserve(bytes);
}

ras::WriteSpan ChunkWriter::reserve_exact(std::size_t bytes) {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    if (!bytes || bytes % impl_->layout.element_size != 0)
        throw ras::Error("Core exact chunk reservation splits an element or is empty");
    impl_->seal();
    impl_->writer = impl_->access->scope->tx->buffer(bytes, impl_->layout);
    impl_->active = true;
    return impl_->writer.reserve(bytes);
}

std::size_t ChunkWriter::size() const {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    const auto current = impl_->active ? impl_->writer.size() : 0;
    if (current > std::numeric_limits<std::size_t>::max() - impl_->output->bytes)
        throw ras::Error("Core chunk output size overflow");
    return impl_->output->bytes + current;
}

ChunkOutput ChunkWriter::finish() {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    impl_->seal();
    impl_->storage->count=impl_->output->count;
    impl_->storage->bytes=impl_->output->bytes;
    impl_->output->storage=detail::BundleAccess::make(impl_->storage);
    const auto descriptor=detail::encode_bundles({impl_->output->storage});
    auto writer=impl_->access->scope->tx->buffer(descriptor.size(),{detail::metadata_schema,1,1});
    auto bytes=writer.reserve(descriptor.size());
    std::memcpy(bytes.bytes().data(),descriptor.data(),descriptor.size());bytes.commit();
    impl_->access->scope->tx->directory_put(impl_->storage->directory,{impl_->storage->id,detail::maximum},{writer.finish(),{},{}});
    detail::grant_bundle(*impl_->access,impl_->output->storage);
    impl_->output->finished = true;
    return ChunkOutput(impl_->output);
}

void ChunkWriter::append(const ras::Buffer& buffer) {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    detail::require_granted(*impl_->access, buffer);
    if (!buffer || !buffer.size() || buffer.layout() != impl_->layout)
        throw ras::Error("appended receipt chunk has incompatible layout");
    impl_->seal();
    if (buffer.size() > std::numeric_limits<std::size_t>::max() - impl_->output->bytes ||
        impl_->output->count == std::numeric_limits<std::size_t>::max())
        throw ras::Error("Core chunk output size overflow");
    impl_->access->scope->tx->directory_put(impl_->storage->directory,
        {impl_->storage->id, impl_->output->count}, {{}, buffer, {}});
    detail::count_block_write(impl_->access->scope->counters.get());
    ++impl_->output->count;
    impl_->output->bytes += buffer.size();
}

void ChunkWriter::seal() {
    if (!impl_) throw ras::Error("empty Core chunk writer");
    impl_->check();
    impl_->seal();
}

} // namespace flexeva::core::runtime
