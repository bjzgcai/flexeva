#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <flexeva/maya/paper_runtime.hpp>
#include <flexeva/maya/paper_cost.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
namespace paper = flexeva::maya::paper;
namespace maya = flexeva::maya;
namespace core = flexeva::core::runtime;
using flexeva::ras::Error;
constexpr const char* session_name = "flexeva.maya.paper.Session";
constexpr const char* snapshot_name = "flexeva.maya.paper.Snapshot";
constexpr const char* cost_basis = "artifact-deterministic-v1-native-v2";
struct PythonError {};
struct Ref {
    PyObject* p = nullptr;
    explicit Ref(PyObject* value = nullptr) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    PyObject* release() { auto* result = p; p = nullptr; return result; }
};
void require(bool ok, const char* message) { if (!ok) throw Error(message); }
PyObject* check(PyObject* value) { if (!value) throw PythonError{}; return value; }
PyObject* field(PyObject* object, const char* name, bool optional = false) {
    require(PyDict_Check(object), "description and records must be dictionaries");
    auto* value = PyDict_GetItemString(object, name);
    require(value || optional, (std::string("missing field: ") + name).c_str());
    return value;
}
std::uint64_t integer(PyObject* object) {
    require(object && PyLong_Check(object) && !PyBool_Check(object), "expected unsigned integer");
    auto value = PyLong_AsUnsignedLongLong(object);
    if (PyErr_Occurred()) throw PythonError{};
    return value;
}
std::uint32_t rank_value(PyObject* object) {
    auto n = integer(object);
    require(n <= std::numeric_limits<std::uint32_t>::max(), "rank is outside uint32 range");
    return static_cast<std::uint32_t>(n);
}
paper::Evidence evidence(PyObject* object) {
    require(PyBytes_Check(object), "semantic evidence must be bytes");
    auto* begin = reinterpret_cast<const std::byte*>(PyBytes_AS_STRING(object));
    return {begin, begin + PyBytes_GET_SIZE(object)};
}
void put(PyObject* object, const char* name, PyObject* value) {
    Ref owned(check(value));
    if (PyDict_SetItemString(object, name, owned.p)) throw PythonError{};
}
void number(PyObject* object, const char* name, std::uint64_t value) {
    put(object, name, PyLong_FromUnsignedLongLong(value));
}
void seconds(PyObject* object, const char* name, double value) {
    put(object, name, PyFloat_FromDouble(value));
}
struct SessionOwner {
    std::unique_ptr<paper::Session> value;
    PyObject* ground_definitions = nullptr;
    bool evaluating = false;
    ~SessionOwner() { value.reset(); Py_XDECREF(ground_definitions); }
};
struct EvaluationGuard {
    SessionOwner& owner;
    explicit EvaluationGuard(SessionOwner& value) : owner(value) {
        require(!owner.evaluating, "Maya session cannot be evaluated recursively");
        owner.evaluating = true;
    }
    ~EvaluationGuard() { owner.evaluating = false; }
};
struct SnapshotOwner { std::shared_ptr<SessionOwner> session; paper::Snapshot value; };
using SessionHandle = std::shared_ptr<SessionOwner>;
SessionHandle& session(PyObject* object) {
    auto* handle = static_cast<SessionHandle*>(PyCapsule_GetPointer(object, session_name));
    if (!handle) throw PythonError{};
    return *handle;
}
SnapshotOwner& snapshot(PyObject* object, const SessionHandle& owner) {
    if (PyDict_Check(object)) object = field(object, "handle");
    auto* result = static_cast<SnapshotOwner*>(PyCapsule_GetPointer(object, snapshot_name));
    if (!result) throw PythonError{};
    require(result->session == owner, "snapshot belongs to another Maya session");
    return *result;
}
void destroy_session(PyObject* capsule) {
    delete static_cast<SessionHandle*>(PyCapsule_GetPointer(capsule, session_name));
}
void destroy_snapshot(PyObject* capsule) {
    delete static_cast<SnapshotOwner*>(PyCapsule_GetPointer(capsule, snapshot_name));
}
template<class F> PyObject* translated(F&& call) {
    try { return call(); }
    catch (const PythonError&) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Python callback failed"); }
    catch (const std::exception& error) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, error.what()); }
    catch (...) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "unknown native exception"); }
    return nullptr;
}

// A Python view owns its immutable RAS backing, never an evaluator pointer or
// a live Torch tensor. Reading a handler input does not copy its numeric body.
struct BufferOwner {
    PyObject_HEAD
    flexeva::ras::ByteView* view;
};
PyTypeObject* buffer_type = nullptr;
int get_buffer(PyObject* object, Py_buffer* buffer, int flags) {
    auto* owner = reinterpret_cast<BufferOwner*>(object);
    if (!owner->view) { PyErr_SetString(PyExc_BufferError, "expired RAS buffer"); return -1; }
    const auto bytes = owner->view->span();
    return PyBuffer_FillInfo(buffer, object, const_cast<std::byte*>(bytes.data()),
                             static_cast<Py_ssize_t>(bytes.size()), 1, flags);
}
void destroy_buffer(PyObject* object) {
    delete reinterpret_cast<BufferOwner*>(object)->view;
    auto* type = Py_TYPE(object);
    type->tp_free(object);
    Py_DECREF(type);
}
PyObject* buffer_view(const flexeva::ras::Buffer& value) {
    auto* owner = PyObject_New(BufferOwner, buffer_type);
    if (!owner) throw PythonError{};
    owner->view = nullptr;
    Ref keep(reinterpret_cast<PyObject*>(owner));
    owner->view = new flexeva::ras::ByteView(value.bytes());
    return check(PyMemoryView_FromObject(keep.p));
}
PyObject* evidence_bytes(const paper::Evidence& bytes) {
    return check(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}
struct BorrowedBuffer {
    Py_buffer value{};
    explicit BorrowedBuffer(PyObject* object) {
        if (PyObject_GetBuffer(object, &value, PyBUF_CONTIG_RO)) throw PythonError{};
    }
    ~BorrowedBuffer() { PyBuffer_Release(&value); }
    std::span<const std::byte> bytes() const {
        return {static_cast<const std::byte*>(value.buf), static_cast<std::size_t>(value.len)};
    }
};
PyObject* ground_request_dict(const core::GroundRequest& request) {
    Ref result(check(PyDict_New()));
    number(result.p, "request_id", request.request_id);
    number(result.p, "point_id", request.point_id);
    Ref lanes(check(PyTuple_New(request.requested_lanes.size())));
    for (std::size_t i = 0; i < request.requested_lanes.size(); ++i)
        PyTuple_SET_ITEM(lanes.p, i, check(PyLong_FromUnsignedLongLong(request.requested_lanes[i])));
    put(result.p, "requested_lanes", lanes.release());
    put(result.p, "query", evidence_bytes(request.query));
    put(result.p, "inputs", buffer_view(request.inputs.payload));
    put(result.p, "input_semantic", evidence_bytes(request.inputs.semantic));
    return result.release();
}
std::vector<core::GroundRegistration> ground_registrations(PyObject* definitions) {
    std::vector<core::GroundRegistration> result;
    Ref rows(check(PySequence_Fast(definitions, "grounds must be a sequence")));
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows.p); ++i) {
        auto* row = PySequence_Fast_GET_ITEM(rows.p, i);
        core::GroundRegistration rule;
        rule.id = integer(field(row, "id")); rule.layer = integer(field(row, "layer"));
        rule.identity = evidence(field(row, "identity"));
        rule.result_schema = integer(field(row, "result_schema"));
        auto* matcher = field(row, "matches"); auto* handler = field(row, "handler");
        require(PyCallable_Check(matcher) && PyCallable_Check(handler), "Ground matcher and handler must be callable");
        rule.matches = [matcher](const core::GroundRequest& request, const core::GroundContext&) {
            Ref args(ground_request_dict(request)), answer(check(PyObject_CallOneArg(matcher, args.p)));
            require(PyBool_Check(answer.p), "Ground matcher must return bool");
            return answer.p == Py_True;
        };
        const auto schema = rule.result_schema;
        rule.handler = [handler, schema](const core::GroundRequest& request, core::GroundContext& context) {
            Ref args(ground_request_dict(request)), answer(check(PyObject_CallOneArg(handler, args.p)));
            core::GroundResult result;
            result.request_id = integer(field(answer.p, "request_id"));
            BorrowedBuffer payload(field(answer.p, "payload"));
            result.value.payload = context.copy(payload.bytes(), schema);
            result.value.semantic = evidence(field(answer.p, "semantic"));
            if (auto* reusable = field(answer.p, "reusable", true)) {
                require(PyBool_Check(reusable), "Ground reusable must be bool");
                result.reusable = reusable == Py_True;
            }
            return result;
        };
        result.push_back(std::move(rule));
    }
    return result;
}

std::vector<paper::Id> identifiers(PyObject* value, const char* message) {
    std::vector<paper::Id> result;
    if (!value) return result;
    Ref rows(check(PySequence_Fast(value, message)));
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows.p); ++i)
        result.push_back(integer(PySequence_Fast_GET_ITEM(rows.p, i)));
    return result;
}
paper::Region parse_region(PyObject* row) {
    paper::Region result;
    result.id = integer(field(row, "id"));
    result.rank = rank_value(field(row, "rank"));
    result.semantic = evidence(field(row, "semantic"));
    if (auto* value = field(row, "revision", true)) result.revision = integer(value);
    if (auto* value = field(row, "execution_region", true)) result.execution_region = integer(value);
    if (auto* value = field(row, "site_id", true)) result.site_id = integer(value);
    if (auto* value = field(row, "site_identity", true)) result.site_identity = evidence(value);
    if (auto* value = field(row, "occurrence_identity", true)) result.occurrence_identity = evidence(value);
    const auto boolean = [&](const char* name, bool& output) {
        if (auto* value = field(row, name, true)) {
            require(PyBool_Check(value), (std::string(name) + " must be boolean").c_str());
            output = value == Py_True;
        }
    };
    boolean("produces_trace", result.produces_trace);
    boolean("semantic_complete", result.semantic_complete);
    boolean("reuse_allowed", result.reuse_allowed);
    return result;
}
// A failed Ground aborts the native transaction. Keep its Python exception
// across workload finally blocks that may attempt further native cleanup.
struct PythonFailure {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    ~PythonFailure() { Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback); }
    void remember() {
        if (type || !PyErr_Occurred()) return;
        PyErr_Fetch(&type, &value, &traceback);
        Py_XINCREF(type); Py_XINCREF(value); Py_XINCREF(traceback);
        PyErr_Restore(type, value, traceback);
    }
    bool restore() {
        if (!type) return false;
        PyErr_Restore(type, value, traceback);
        type = value = traceback = nullptr;
        return true;
    }
};
struct Sink {
    PyObject_HEAD
    paper::Capture* capture;
    const paper::Input* input;
    std::map<paper::Id, std::uint32_t>* declared;
    PythonFailure* failure;
};
PyTypeObject* sink_type = nullptr;
paper::Capture& active(Sink* self) {
    require(self->capture && self->input, "native capture scope has expired");
    return *self->capture;
}
PyObject* identity(Sink* self, PyObject* args) {
    return translated([&]() -> PyObject* {
        unsigned long long domain;
        PyObject* coordinate;
        if (!PyArg_ParseTuple(args, "KO", &domain, &coordinate)) throw PythonError{};
        BorrowedBuffer bytes(coordinate);
        return check(PyLong_FromUnsignedLongLong(active(self).identity(domain, bytes.bytes())));
    });
}
PyObject* declare_region(Sink* self, PyObject* row) {
    return translated([&]() -> PyObject* {
        auto& capture = active(self);
        const auto region = parse_region(row);
        const auto inputs = identifiers(field(row, "inputs", true), "producer IDs must be a sequence");
        capture.declare_region(region, inputs);
        require(self->declared != nullptr, "native Code writer has expired");
        const auto [found, inserted] = self->declared->emplace(region.id, region.rank);
        require(inserted || found->second == region.rank, "Code occurrence changed its actual rank");
        Py_RETURN_NONE;
    });
}
PyObject* begin_region(Sink* self, PyObject* args) {
    return translated([&]() -> PyObject* {
        unsigned long long id; PyObject* rank = nullptr;
        if (!PyArg_ParseTuple(args, "K|O", &id, &rank)) throw PythonError{};
        auto& capture = active(self);
        if (rank) {
            const auto expected = rank_value(rank);
            const auto it = self->declared->find(id);
            require(it != self->declared->end() && it->second == expected,
                    "region rank does not match its Code declaration");
        }
        capture.begin_region(id); Py_RETURN_NONE;
    });
}
PyObject* end_region(Sink* self, PyObject*) {
    return translated([&]() -> PyObject* { active(self).end_region(); Py_RETURN_NONE; });
}
PyObject* recording(Sink* self, PyObject*) {
    return translated([&]() -> PyObject* { return PyBool_FromLong(active(self).recording()); });
}
PyObject* communicator(Sink* self, PyObject* args) {
    return translated([&]() -> PyObject* {
        unsigned long long handle; PyObject *membership, *local;
        if (!PyArg_ParseTuple(args, "KOO", &handle, &membership, &local)) throw PythonError{};
        auto& capture = active(self);
        paper::Id group = 0;
        if (PyLong_Check(membership)) group = integer(membership);
        else {
            Ref seq(check(PySequence_Fast(membership, "communicator members must be a sequence")));
            std::vector<std::uint32_t> members;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq.p); ++i)
                members.push_back(rank_value(PySequence_Fast_GET_ITEM(seq.p, i)));
            for (const auto& candidate : self->input->communicators) if (candidate.members == members) {
                require(!group, "ambiguous communicator membership; pass explicit group ID");
                group = candidate.id;
            }
            require(group != 0, "communicator membership is not declared");
        }
        capture.communicator(handle, group, rank_value(local)); Py_RETURN_NONE;
    });
}
PyObject* ground(Sink* self, PyObject* argument) {
    auto* result = translated([&]() -> PyObject* {
        auto& capture = active(self);
        core::GroundRequest request;
        request.request_id = integer(field(argument, "request_id"));
        request.point_id = integer(field(argument, "point_id"));
        request.query = evidence(field(argument, "query"));
        request.inputs.semantic = evidence(field(argument, "input_semantic"));
        Ref lanes(check(PySequence_Fast(field(argument, "requested_lanes"), "requested_lanes must be a sequence")));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(lanes.p); ++i)
            request.requested_lanes.push_back(integer(PySequence_Fast_GET_ITEM(lanes.p, i)));
        const auto schema = integer(field(argument, "input_schema"));
        BorrowedBuffer input(field(argument, "inputs"));
        const auto resolution = capture.ground(std::move(request), input.bytes(), schema);
        Ref result(check(PyDict_New()));
        number(result.p, "request_id", resolution.request_id);
        number(result.p, "fact_id", resolution.fact_id);
        put(result.p, "reusable", PyBool_FromLong(resolution.reusable));
        put(result.p, "payload", buffer_view(resolution.value.payload));
        put(result.p, "semantic", evidence_bytes(resolution.value.semantic));
        return result.release();
    });
    if (!result && self->failure) self->failure->remember();
    return result;
}
PyObject* code_occurrence(Sink* self, PyObject* args) {
    return translated([&]() -> PyObject* {
        unsigned long long region;
        PyObject *packet, *facts = nullptr, *inputs = nullptr;
        if (!PyArg_ParseTuple(args, "KO|OO", &region, &packet, &facts, &inputs)) throw PythonError{};
        BorrowedBuffer bytes(packet);
        const auto ground_facts = identifiers(facts, "Ground fact IDs must be a sequence");
        const auto dependency_refs = identifiers(inputs, "producer IDs must be a sequence");
        active(self).code_occurrence(region, bytes.bytes(), ground_facts, dependency_refs);
        Py_RETURN_NONE;
    });
}
PyObject* code_dictionary(Sink* self, PyObject* packet) {
    return translated([&]() -> PyObject* {
        BorrowedBuffer bytes(packet);
        active(self).code_dictionary(bytes.bytes());
        Py_RETURN_NONE;
    });
}
PyMethodDef sink_methods[] = {
    {"identity", reinterpret_cast<PyCFunction>(identity), METH_VARARGS, nullptr},
    {"declare_region", reinterpret_cast<PyCFunction>(declare_region), METH_O, nullptr},
    {"begin_region", reinterpret_cast<PyCFunction>(begin_region), METH_VARARGS, nullptr},
    {"end_region", reinterpret_cast<PyCFunction>(end_region), METH_NOARGS, nullptr},
    {"recording", reinterpret_cast<PyCFunction>(recording), METH_NOARGS, nullptr},
    {"communicator", reinterpret_cast<PyCFunction>(communicator), METH_VARARGS, nullptr},
    {"ground", reinterpret_cast<PyCFunction>(ground), METH_O, nullptr},
    {"code_occurrence", reinterpret_cast<PyCFunction>(code_occurrence), METH_VARARGS, nullptr},
    {"code_dictionary", reinterpret_cast<PyCFunction>(code_dictionary), METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

paper::Input parse_input(PyObject* description, PyObject* execute) {
    require(PyCallable_Check(execute), "workload executor must be callable");
    paper::Input input;
    if (auto* dynamic = field(description, "dynamic_regions", true)) {
        require(PyBool_Check(dynamic), "dynamic_regions must be boolean");
        input.dynamic_regions = dynamic == Py_True;
    }
    input.configuration = evidence(field(description, "configuration"));
    if (auto* value = field(description, "window_semantic", true))
        input.window_semantic = evidence(value);
    if (auto* value = field(description, "window_complete", true)) {
        require(PyBool_Check(value), "window_complete must be boolean");
        input.window_complete = value == Py_True;
    }
    if (auto* provenance = field(description, "frontend_evidence", true))
        input.frontend_evidence = evidence(provenance);
    if (auto* values = field(description, "regions", input.dynamic_regions)) {
        Ref regions(check(PySequence_Fast(values, "regions must be a sequence")));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(regions.p); ++i)
            input.regions.push_back(parse_region(PySequence_Fast_GET_ITEM(regions.p, i)));
    }
    if (auto* dependencies = field(description, "dependencies", true)) {
        Ref values(check(PySequence_Fast(dependencies, "dependencies must be a sequence")));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(values.p); ++i) {
            Ref edge(check(PySequence_Fast(PySequence_Fast_GET_ITEM(values.p, i), "dependency must be a pair")));
            require(PySequence_Fast_GET_SIZE(edge.p) == 2, "dependency must have two endpoints");
            input.dependencies.push_back({integer(PySequence_Fast_GET_ITEM(edge.p, 0)), integer(PySequence_Fast_GET_ITEM(edge.p, 1))});
        }
    }
    if (auto* groups = field(description, "communicators", true)) {
        Ref values(check(PySequence_Fast(groups, "communicators must be a sequence")));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(values.p); ++i) {
            auto* row = PySequence_Fast_GET_ITEM(values.p, i);
            paper::Communicator group; group.id = integer(field(row, "id"));
            Ref members(check(PySequence_Fast(field(row, "members"), "members must be a sequence")));
            for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(members.p); ++j)
                group.members.push_back(rank_value(PySequence_Fast_GET_ITEM(members.p, j)));
            input.communicators.push_back(std::move(group));
        }
    }
    if (auto* projections = field(description, "rank_projections", true)) {
        Ref rows(check(PySequence_Fast(projections, "rank_projections must be a sequence")));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows.p); ++i) {
            auto* row = PySequence_Fast_GET_ITEM(rows.p, i);
            paper::RankProjection projection;
            projection.rank = rank_value(field(row, "rank"));
            projection.representative = rank_value(field(row, "representative"));
            if (auto* mapping = field(row, "communicator_map", true)) {
                Ref pairs(check(PySequence_Fast(mapping, "communicator_map must be a sequence")));
                for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(pairs.p); ++j) {
                    Ref pair(check(PySequence_Fast(PySequence_Fast_GET_ITEM(pairs.p, j), "communicator mapping must be a pair")));
                    require(PySequence_Fast_GET_SIZE(pair.p) == 2, "communicator mapping must have two endpoints");
                    projection.communicator_map.emplace_back(integer(PySequence_Fast_GET_ITEM(pair.p, 0)),
                                                              integer(PySequence_Fast_GET_ITEM(pair.p, 1)));
                }
            }
            input.rank_projections.push_back(std::move(projection));
        }
    }
    return input;
}
void set_executor(paper::Input& input, PyObject* callable) {
    input.execute = [&input, callable](paper::Capture& capture) {
        PythonFailure failure;
        std::map<paper::Id, std::uint32_t> declared;
        for (const auto& region : input.regions) declared.emplace(region.id, region.rank);
        auto* sink = PyObject_New(Sink, sink_type);
        if (!sink) throw PythonError{};
        Ref owner(reinterpret_cast<PyObject*>(sink));
        sink->capture = &capture; sink->input = &input; sink->declared = &declared;
        sink->failure = &failure;
        Ref result(PyObject_CallOneArg(callable, owner.p));
        sink->capture = nullptr; sink->input = nullptr; sink->declared = nullptr;
        sink->failure = nullptr;
        if (failure.restore()) throw PythonError{};
        try { capture.check_native_failure(); }
        catch (...) { PyErr_Clear(); throw; }
        check(result.p);
    };
}

maya::Cost native_cost(const FakeCudaRasRecordV2& native) {
    const auto event = maya::native_event_cost_input(native);
    // The artifact parser omitted protocol-only NCCL records. Keep them in the
    // scheduling graph with zero cost; the pure artifact cost formula is unchanged.
    const auto us = event.kind == "other" ? 0 : maya::predict_event_us(event);
    require(std::isfinite(us) && us >= 0 && us * 1000 < static_cast<double>(std::numeric_limits<maya::Tick>::max()), "cost outside tick range");
    const auto ticks = static_cast<maya::Tick>(std::llround(us * 1000));
    const bool device = event.kind == "kernel_launch" || event.kind == "blas_compute" ||
                        event.kind == "nccl_collective" || event.kind == "mem_copy";
    return device ? maya::Cost{0, ticks} : maya::Cost{ticks, 0};
}
paper::Evidence native_cost_key(const FakeCudaRasRecordV2& native) {
    const auto event = maya::native_event_cost_input(native);
    paper::Evidence result;
    const auto number = [&](std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) {
            result.push_back(static_cast<std::byte>(value & 255));
            value >>= 8;
        }
    };
    const auto string = [&](std::string_view value) {
        number(value.size());
        for (unsigned char c : value) result.push_back(static_cast<std::byte>(c));
    };
    string(event.api); string(event.kind);
    number(event.stream); number(event.bytes); number(event.count);
    number(static_cast<std::uint64_t>(event.rank)); number(event.blocking);
    // predict_event_us ignores duration hints. Rank is retained because the
    // artifact forest uses it even for otherwise identical native bodies.
    return result;
}
PyObject* memory_dict(const flexeva::ras::MemoryStats& memory) {
    Ref result(check(PyDict_New()));
#define MEMORY_FIELD(name) number(result.p, #name, memory.name)
    MEMORY_FIELD(payload_bytes); MEMORY_FIELD(capacity_bytes); MEMORY_FIELD(index_bytes);
    MEMORY_FIELD(metadata_bytes); MEMORY_FIELD(blocks); MEMORY_FIELD(index_pages);
    MEMORY_FIELD(versions); MEMORY_FIELD(pending_writers); MEMORY_FIELD(pending_reservations);
    MEMORY_FIELD(metadata_array_bytes); MEMORY_FIELD(live_payload_allocations);
    MEMORY_FIELD(peak_capacity_bytes); MEMORY_FIELD(allocated_payload_bytes);
    MEMORY_FIELD(copied_payload_bytes); MEMORY_FIELD(created_index_pages);
    MEMORY_FIELD(view_requests); MEMORY_FIELD(allocation_requests);
    MEMORY_FIELD(reused_payload_slots); MEMORY_FIELD(copied_metadata_bytes);
    MEMORY_FIELD(index_page_visits); MEMORY_FIELD(publish_count);
#undef MEMORY_FIELD
    return result.release();
}
PyObject* snapshot_dict(const SessionHandle& owner, paper::Snapshot value) {
    Ref result(check(PyDict_New()));
    Ref feedback(check(PyDict_New()));
    number(feedback.p, "makespan_ns", value.feedback.makespan);
    put(feedback.p, "cost_basis", PyUnicode_FromString(cost_basis));
    Ref ranks(check(PyDict_New()));
    for (auto [rank, end] : value.feedback.rank_end) {
        Ref key(check(PyLong_FromUnsignedLong(rank))), duration(check(PyLong_FromUnsignedLongLong(end)));
        if (PyDict_SetItem(ranks.p, key.p, duration.p)) throw PythonError{};
    }
    put(feedback.p, "rank_end_ns", ranks.release());
    put(result.p, "feedback", feedback.release());
    Ref metrics(check(PyDict_New()));
    const auto& m = value.metrics;
    put(metrics.p, "diagnostics_enabled", PyBool_FromLong(m.diagnostics_enabled));
#define METRIC(name) number(metrics.p, #name, m.name)
    METRIC(native_calls); METRIC(retained_calls); METRIC(retained_trace_calls);
    METRIC(retained_record_checks);
    METRIC(normalization_raw_records); METRIC(normalized_output_adoptions);
    METRIC(retained_normalized_body_reads); METRIC(publication_body_reads);
    METRIC(publication_descriptor_visits); METRIC(feedback_body_reads);
    METRIC(raw_records); METRIC(raw_bytes);
    METRIC(copied_trace_bytes); METRIC(selected_regions); METRIC(executed_regions);
    METRIC(preparation_regions); METRIC(trace_partitions); METRIC(trace_partitions_read);
    METRIC(cost_calls); METRIC(scheduled_events); METRIC(published_versions);
    METRIC(replay_input_reads); METRIC(replay_graph_nodes); METRIC(replay_updated_nodes);
    METRIC(replay_reused_nodes); METRIC(replay_graph_rebuilt);
#undef METRIC
    number(metrics.p, "resource_records_read", m.resource_records_read);
    seconds(metrics.p, "capture_seconds", m.capture_seconds); seconds(metrics.p, "cost_seconds", m.cost_seconds);
    seconds(metrics.p, "capture_setup_seconds", m.capture_setup_seconds);
    seconds(metrics.p, "host_execution_seconds", m.host_execution_seconds);
    seconds(metrics.p, "capture_finalize_seconds", m.capture_finalize_seconds);
    seconds(metrics.p, "normalization_seconds", m.normalization_seconds);
    seconds(metrics.p, "retained_state_seconds", m.retained_state_seconds);
    seconds(metrics.p, "replay_seconds", m.replay_seconds); seconds(metrics.p, "total_seconds", m.total_seconds);
    put(metrics.p, "memory", memory_dict(m.memory));
    Ref core(check(PyDict_New()));
#define CORE_FIELD(name) number(core.p, #name, m.core.name)
    CORE_FIELD(payload_reads); CORE_FIELD(full_enumerations); CORE_FIELD(metadata_reads);
    CORE_FIELD(selected_boundaries); CORE_FIELD(generated_receipts); CORE_FIELD(dependency_visits);
    CORE_FIELD(partition_callbacks); CORE_FIELD(boundary_callbacks); CORE_FIELD(apply_callbacks);
    CORE_FIELD(link_callbacks); CORE_FIELD(state_callbacks);
    CORE_FIELD(header_reads); CORE_FIELD(selection_evidence_bytes);
    CORE_FIELD(lineage_upserts); CORE_FIELD(lineage_removals); CORE_FIELD(lineage_retained);
    CORE_FIELD(boundary_input_writes);
    CORE_FIELD(merge_compare_bytes);
    CORE_FIELD(chunk_intern_hash_bytes); CORE_FIELD(chunk_intern_compare_bytes);
    CORE_FIELD(chunk_intern_shared_bytes); CORE_FIELD(chunk_intern_reused_blocks);
    CORE_FIELD(ground_calls); CORE_FIELD(ground_operation_hits); CORE_FIELD(ground_parent_hits);
    CORE_FIELD(prepared_adoptions); CORE_FIELD(previous_output_adoptions);
    CORE_FIELD(shared_output_adoptions); CORE_FIELD(output_root_bindings); CORE_FIELD(output_block_writes);
    CORE_FIELD(route_upserts); CORE_FIELD(route_removals); CORE_FIELD(route_visits);
    CORE_FIELD(route_entries); CORE_FIELD(materialized_output_bindings);
    CORE_FIELD(admission_body_bytes_read); CORE_FIELD(admission_metadata_bytes_read);
    CORE_FIELD(adoption_body_bytes_read); CORE_FIELD(adopted_block_index_writes);
#undef CORE_FIELD
    seconds(core.p, "selection_seconds", m.core.selection_seconds);
    seconds(core.p, "partition_seconds", m.core.partition_seconds);
    seconds(core.p, "lineage_seconds", m.core.lineage_seconds);
    seconds(core.p, "ground_seconds", m.core.ground_seconds);
    put(core.p, "diagnostics_enabled", PyBool_FromLong(m.core.diagnostics_enabled));
    seconds(core.p, "evaluate_seconds", m.core.evaluate_seconds);
    seconds(core.p, "evaluator_callback_seconds", m.core.evaluator_callback_seconds);
    seconds(core.p, "framework_seconds", m.core.framework_seconds);
    put(metrics.p, "core", core.release());
    put(result.p, "metrics", metrics.release());
    auto handle = std::make_unique<SnapshotOwner>(SnapshotOwner{owner, std::move(value)});
    Ref capsule(check(PyCapsule_New(handle.get(), snapshot_name, destroy_snapshot)));
    handle.release();
    put(result.p, "handle", capsule.release());
    return result.release();
}
PyObject* new_session(PyObject*, PyObject* args, PyObject* kwargs) {
    return translated([&]() -> PyObject* {
        const char* mode = "refresh"; unsigned long long chunk = paper::Options{}.chunk_bytes;
        PyObject* grounds = Py_None;
        int diagnostics = 0;
        static const char* names[] = {"mode", "chunk_bytes", "grounds", "diagnostics", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sKOp", const_cast<char**>(names), &mode, &chunk, &grounds, &diagnostics)) throw PythonError{};
        paper::Options options; const std::string selected(mode);
        if (selected == "refresh") options.mode = paper::Mode::Refresh;
        else if (selected == "full-rank") options.mode = paper::Mode::FullRank;
        else if (selected == "full-compact") options.mode = paper::Mode::FullCompact;
        else throw Error("unknown evaluation mode");
        options.chunk_bytes = chunk; options.native_costs = native_cost;
        options.diagnostics = diagnostics != 0;
        options.native_cost_key = native_cost_key;
        constexpr std::string_view cost_id = "artifact-deterministic-v1-native-v2-ns";
        options.cost_identity.assign(reinterpret_cast<const std::byte*>(cost_id.data()), reinterpret_cast<const std::byte*>(cost_id.data()+cost_id.size()));
        auto owner = std::make_shared<SessionOwner>();
        if (grounds != Py_None) {
            Ref definitions(check(PySequence_Fast(grounds, "grounds must be a sequence")));
            Ref frozen(check(PyTuple_New(PySequence_Fast_GET_SIZE(definitions.p))));
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(definitions.p); ++i) {
                auto* row = PySequence_Fast_GET_ITEM(definitions.p, i);
                require(PyDict_Check(row), "Ground registration must be a dictionary");
                PyTuple_SET_ITEM(frozen.p, i, check(PyDict_Copy(row)));
            }
            owner->ground_definitions = frozen.release();
            options.grounds = ground_registrations(owner->ground_definitions);
        }
        owner->value = std::make_unique<paper::Session>(std::move(options));
        auto handle = std::make_unique<SessionHandle>(std::move(owner));
        Ref capsule(check(PyCapsule_New(handle.get(), session_name, destroy_session)));
        handle.release(); return capsule.release();
    });
}
PyObject* init(PyObject*, PyObject* args) {
    return translated([&]() -> PyObject* {
        PyObject *handle, *description, *execute;
        if (!PyArg_ParseTuple(args, "OOO", &handle, &description, &execute)) throw PythonError{};
        auto owner = session(handle); auto input = parse_input(description, execute); set_executor(input, execute);
        EvaluationGuard guard(*owner);
        return snapshot_dict(owner, owner->value->init(input));
    });
}
PyObject* refresh(PyObject*, PyObject* args) {
    return translated([&]() -> PyObject* {
        PyObject *handle, *parent, *description, *execute;
        if (!PyArg_ParseTuple(args, "OOOO", &handle, &parent, &description, &execute)) throw PythonError{};
        auto owner = session(handle); auto& previous = snapshot(parent, owner);
        auto input = parse_input(description, execute); set_executor(input, execute);
        EvaluationGuard guard(*owner);
        return snapshot_dict(owner, owner->value->refresh(previous.value, input));
    });
}
PyObject* commit(PyObject*, PyObject* args) {
    return translated([&]() -> PyObject* {
        PyObject *handle, *state;
        if (!PyArg_ParseTuple(args, "OO", &handle, &state)) throw PythonError{};
        auto owner = session(handle); return snapshot_dict(owner, owner->value->commit(snapshot(state, owner).value));
    });
}
PyObject* inspect(PyObject*, PyObject* args) {
    return translated([&]() -> PyObject* {
        PyObject *handle, *state;
        if (!PyArg_ParseTuple(args, "OO", &handle, &state)) throw PythonError{};
        auto owner = session(handle); const auto result = owner->value->inspect(snapshot(state, owner).value);
        Ref output(check(PyDict_New()));
        put(output.p, "normalized_digest", PyUnicode_FromString(result.normalized_digest.c_str()));
        put(output.p, "dependency_digest", PyUnicode_FromString(result.dependency_digest.c_str()));
        number(output.p, "logical_events", result.logical_events); number(output.p, "physical_records", result.physical_records);
        number(output.p, "trace_bytes", result.trace_bytes); return output.release();
    });
}
PyObject* memory(PyObject*, PyObject* args) {
    return translated([&]() -> PyObject* {
        PyObject* handle; if (!PyArg_ParseTuple(args, "O", &handle)) throw PythonError{};
        return memory_dict(session(handle)->value->store().stats());
    });
}
PyMethodDef module_methods[] = {
    {"new_session", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(new_session)), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"init", init, METH_VARARGS, nullptr}, {"refresh", refresh, METH_VARARGS, nullptr},
    {"commit", commit, METH_VARARGS, nullptr}, {"inspect", inspect, METH_VARARGS, nullptr},
    {"memory", memory, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}
};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_flexeva_maya_paper", nullptr, -1, module_methods, nullptr, nullptr, nullptr, nullptr};
}

PyMODINIT_FUNC PyInit__flexeva_maya_paper() {
    PyType_Slot slots[] = {{Py_tp_methods, sink_methods}, {0, nullptr}};
    PyType_Spec spec = {"_flexeva_maya_paper.Capture", sizeof(Sink), 0, Py_TPFLAGS_DEFAULT, slots};
    Ref type(PyType_FromSpec(&spec));
    if (!type.p) return nullptr;
    sink_type = reinterpret_cast<PyTypeObject*>(type.p);
    Ref module(PyModule_Create(&definition));
    if (!module.p) return nullptr;
    if (PyModule_AddObjectRef(module.p, "Capture", type.p)) return nullptr;
    PyType_Slot buffer_slots[] = {
        {Py_bf_getbuffer, reinterpret_cast<void*>(get_buffer)},
        {Py_tp_dealloc, reinterpret_cast<void*>(destroy_buffer)}, {0, nullptr}
    };
    PyType_Spec buffer_spec = {"_flexeva_maya_paper.RasBuffer", sizeof(BufferOwner), 0,
                              Py_TPFLAGS_DEFAULT, buffer_slots};
    Ref buffer(PyType_FromSpec(&buffer_spec));
    if (!buffer.p) return nullptr;
    buffer_type = reinterpret_cast<PyTypeObject*>(buffer.p);
    if (PyModule_AddObjectRef(module.p, "_RasBuffer", buffer.p)) return nullptr;
    return module.release();
}
