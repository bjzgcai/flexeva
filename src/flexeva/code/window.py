"""Closed Code windows built from evaluator-bound dynamic emissions.

Runtime hooks are construction inputs, not RAS events. A collector folds host
hooks into one block between actual emissions, retaining portable provenance
and dependency evidence rather than live operands or an execution journal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..errors import PipelineContractError
from ..model import (
    Event,
    EventCapture,
    EventDependency,
    EventPayload,
    LaneContext,
    NeedGround,
    PayloadBinding,
    SourceSpan,
    _canonical,
    stable_digest,
)
from ..payloads import backing_context, backing_value
from .cfg import CodeControlFlow
from .dynamic import DynamicCodeProgram, dynamic_program_envelope, restore_dynamic_program
from .dynamic_dependencies import DynamicDependencyTracker
from .emission import checked_contract_identity
from .inputs import DynamicInputEvidence, safe_input
from .occurrence import OccurrenceTracker
from .rank import RankTracker
from .taint import CodeTaintPolicy, checked_analysis_identity

if TYPE_CHECKING:
    from flexeva_workload_runtime import WorkloadInvocation


@dataclass(frozen=True)
class CodeSourceFragment:
    """One actually reached source site, possibly reached repeatedly in a block."""

    site_id: str
    owner: str
    span: SourceSpan
    source_text: str
    normalized: str
    callee: str | None
    role: str
    defs: tuple[str, ...]
    uses: tuple[str, ...]
    count: int = 1


@dataclass(frozen=True)
class CodeOccurrence:
    event_id: str
    lane: int | str
    lane_order: int
    inputs: DynamicInputEvidence
    execution_digest: str


@dataclass(frozen=True)
class CodeWindowEventData:
    fragments: tuple[CodeSourceFragment, ...]
    execution_count: int
    region_id: str
    region_activation: Any
    region_occurrence: int
    inputs: DynamicInputEvidence
    emission_binding: Mapping[str, str] | None
    structure_digest: str
    activation_path: tuple[str, ...]
    rank_fingerprint: str
    occurrences: tuple[CodeOccurrence, ...]
    context: Any = None


@dataclass(frozen=True)
class CodeWindowInput:
    """Per-evaluation frozen program and invocations, never registered policy."""

    program: DynamicCodeProgram
    invocations: tuple[Any, ...]

    def __post_init__(self):
        if not isinstance(self.program, DynamicCodeProgram) or not isinstance(self.invocations, tuple):
            raise PipelineContractError("Code input requires a frozen program and invocation tuple")


@dataclass(frozen=True)
class CodeWindowEvidence:
    program: DynamicCodeProgram
    code_contract_identity: Mapping[str, Any]
    invocations: tuple[Mapping[str, Any], ...]
    analysis_identity: Mapping[str, Any]
    cfg_digest: str
    protocol_version: str = "code-window-v7"


def _fragment(occurrence: Any) -> CodeSourceFragment:
    site = occurrence.site
    span = site.get("span")
    if isinstance(span, Mapping):
        span = SourceSpan(**dict(span))
    if not isinstance(span, SourceSpan):
        raise PipelineContractError("Code source fragment requires exact source provenance")
    for key in ("owner", "normalized", "source_text"):
        if type(site.get(key)) is not str:
            raise PipelineContractError("Code source fragment requires portable source strings")
    callee, role = site.get("callee"), site.get("role", "statement")
    if ((callee is not None and type(callee) is not str) or type(role) is not str
            or type(occurrence.site_id) is not str or not occurrence.site_id
            or any(type(name) is not str for name in (*site.get("defs", ()), *site.get("uses", ())))):
        raise PipelineContractError("Code source fragment metadata is invalid")
    return CodeSourceFragment(
        occurrence.site_id, site["owner"], span, site["source_text"], site["normalized"],
        callee, role, tuple(site.get("defs", ())), tuple(site.get("uses", ())),
    )


def _evidence_digest(site_id, evidence):
    """Same stable digest, without rewalking already checked JSON inputs."""
    value = {**vars(evidence), "state_refs": _canonical(evidence.state_refs),
             "unresolved": _canonical(evidence.unresolved)}
    encoded = json.dumps((site_id, value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().encode("ascii")


class _EventBuilder:
    """A bounded source-site table plus a streaming execution fingerprint."""

    def __init__(self, occurrence: Any, iteration_path=()) -> None:
        if occurrence.kind not in ("block", "emission"):
            raise PipelineContractError("Code windows contain only block and emission events")
        if type(occurrence.lane) not in (int, str):
            raise PipelineContractError("Code lane must be int or str, excluding bool")
        order = occurrence.inputs.get("event_order")
        if type(order) is not int or order < 0:
            raise PipelineContractError("Code construction requires a nonnegative logical event order")
        if (type(occurrence.instance_id) is not str or not occurrence.instance_id
                or type(occurrence.region_id) is not str or not occurrence.region_id):
            raise PipelineContractError("Code construction requires event and region identities")
        self.event_id, self.kind = occurrence.instance_id, occurrence.kind
        self.lane, self.lane_order = occurrence.lane, order
        self.region_id = occurrence.region_id
        self.region_activation = safe_input(occurrence.region_activation)
        self.region_occurrence = occurrence.region_occurrence
        self.fragments: dict[str, CodeSourceFragment] = {}
        self.digest = hashlib.sha256()
        self.structure = hashlib.sha256()
        self.rank_digest = hashlib.sha256()
        self.activation_path = (*occurrence.activation_path, *iteration_path)
        self.count = 0
        self.producers: set[tuple[str, str]] = set()
        self.semantic: dict[str, Any] = {}
        self.sources: dict[str, Any] = {}
        self.refs: dict[str, Any] = {}
        self.unresolved: set[str] = set()
        self.complete = True
        binding = occurrence.inputs.get("emission_binding")
        self.binding = safe_input(binding) if binding is not None else None
        if self.kind == "emission":
            if (type(self.binding) is not dict
                    or set(self.binding) != {"transform_id", "operation_id"}
                    or any(type(value) is not str or not value for value in self.binding.values())):
                raise PipelineContractError("emission requires an evaluator-owned binding")
        elif self.binding is not None:
            raise PipelineContractError("block cannot carry an emission binding")

    def observe(self, occurrence: Any, evidence: DynamicInputEvidence, rank_facts=()) -> None:
        if (occurrence.instance_id != self.event_id or occurrence.kind != self.kind
                or occurrence.lane != self.lane or occurrence.inputs.get("event_order") != self.lane_order):
            raise PipelineContractError("host fragments must share their reserved Code block identity")
        fragment = _fragment(occurrence)
        old = self.fragments.get(fragment.site_id)
        if old is not None:
            if replace(old, count=1) != fragment:
                raise PipelineContractError("a Code source site changed during one window")
            fragment = replace(fragment, count=old.count + 1)
        self.fragments[fragment.site_id] = fragment
        # Producer IDs internal to this block are reaching definitions, not
        # RAS self-edges. The incremental digest still records their execution.
        # The collector already checked/detached this before rank observation.
        self.producers.update(pair for pair in evidence.producer_inputs if pair[0] != self.event_id)
        self.semantic[fragment.site_id] = evidence.semantic_inputs
        self.sources[fragment.site_id] = evidence.source_inputs
        self.refs.update((ref.value, ref) for ref in evidence.state_refs)
        self.unresolved.update(evidence.unresolved)
        self.complete &= evidence.inputs_complete
        self.digest.update(_evidence_digest(fragment.site_id, evidence))
        self.structure.update(stable_digest(fragment.site_id).encode("ascii"))
        self.rank_digest.update(stable_digest((fragment.site_id, rank_facts)).encode("ascii"))
        self.count += 1

    def event(self) -> Event:
        from .sharing import merged_inputs

        fragments = tuple(self.fragments.values())
        evidence = DynamicInputEvidence(
            tuple(sorted(self.producers)), dict(self.semantic), dict(self.sources),
            tuple(self.refs[key] for key in sorted(self.refs)), self.complete,
            tuple(sorted(self.unresolved)),
        ).checked()
        occurrences = (CodeOccurrence(self.event_id, self.lane, self.lane_order,
                                      evidence, self.digest.hexdigest()),)
        raw = CodeWindowEventData(
            fragments, self.count, self.region_id,
            self.region_activation, self.region_occurrence,
            merged_inputs(occurrences, {}, self.event_id), self.binding,
            self.structure.hexdigest(), self.activation_path, self.rank_digest.hexdigest(),
            occurrences,
        )
        return Event(
            self.event_id, self.kind, PayloadBinding.inline(raw), scope_id=self.region_id,
            match_key=self.event_id,
            provenance=fragments[0].span if len(fragments) == 1 else None,
            lane_context=LaneContext((self.lane,)),
            defs=tuple(sorted({name for fragment in fragments for name in fragment.defs})),
            uses=tuple(sorted({name for fragment in fragments for name in fragment.uses})),
        )


@dataclass(frozen=True)
class _EmissionTicket:
    event_id: str


class CodeWindowCollector:
    """Portable event summaries with an execution-local input provider."""

    def __init__(self, *, input_provider: Any = None, program=None, analysis=None, taint_policy=None) -> None:
        self.input_provider = input_provider if input_provider is not None else DynamicDependencyTracker()
        self.events: list[Event] = []
        self._block: _EventBuilder | None = None
        self._pending: dict[str, _EventBuilder] = {}
        self._failed = False
        self._ground_facts = []
        self._body_sources = {item["owner"]: item["source_digest"] for item in program.owned_definitions} if program else {}
        analysis = analysis if analysis is not None else CodeControlFlow(program) if program is not None else None
        self.rank_tracker = RankTracker(analysis, taint_policy)
        self.occurrence_tracker = OccurrenceTracker(analysis)

    def _evidence(self, occurrence: Any) -> DynamicInputEvidence:
        result = self.input_provider(occurrence)
        if not isinstance(result, DynamicInputEvidence):
            raise PipelineContractError("Code input provider must return DynamicInputEvidence")
        return result.checked()

    def _hook(self, name: str, *args: Any, **kwargs: Any) -> Any:
        hook = getattr(self.input_provider, name, None)
        return hook(*args, **kwargs) if hook is not None else None

    def observe_block(self, occurrence: Any, result: Any = None) -> None:
        if occurrence.kind != "block":
            raise PipelineContractError("host construction hook must describe a block fragment")
        self.occurrence_tracker.observe(occurrence)
        if self._block is None:
            self._block = self._builder(occurrence)
        self._block.observe(occurrence, self._evidence(occurrence), self.rank_tracker.observe(occurrence, result))
        for fact in self._ground_facts:
            self._record_fact(self._block, fact)
        self._ground_facts.clear()
        # Complete each microstep now, before owned return/deactivation removes
        # its source scope. All its definitions name the reserved block event.
        self._hook("complete", occurrence, result=result, error=None)

    def flush_block(self) -> None:
        if self._block is not None:
            self.events.append(self._block.event())
            self._block = None

    @staticmethod
    def _record_fact(builder, fact):
        builder.semantic["ground:" + fact["point_id"]] = fact
        digest = stable_digest(fact).encode("ascii")
        builder.digest.update(digest)
        builder.rank_digest.update(digest)

    def record_ground(self, fact):
        if self._pending:
            self._record_fact(next(reversed(self._pending.values())), fact)
        else:
            self._ground_facts.append(fact)

    def record_body_summary(self, ticket, summary):
        builder = self._pending[ticket.event_id]
        owners = sorted(summary["owners"])
        if not owners or any(owner not in self._body_sources for owner in owners):
            raise PipelineContractError("summarized body requires known frozen source definitions")
        fact = {"contract": safe_input(summary["contract"]),
                "sources": tuple((owner, self._body_sources[owner]) for owner in owners)}
        builder.semantic["$body_summary"] = fact
        digest = stable_digest(fact).encode("ascii")
        builder.digest.update(digest)
        builder.rank_digest.update(digest)

    def arrive(self, occurrence: Any) -> _EmissionTicket:
        if occurrence.kind != "emission" or self._block is not None:
            raise PipelineContractError("emission arrival requires a flushed preceding block")
        if occurrence.instance_id in self._pending:
            raise PipelineContractError("duplicate pending Code emission")
        builder = self._builder(occurrence)
        builder.observe(occurrence, self._evidence(occurrence), self.rank_tracker.observe(occurrence, completed=False))
        self._pending[occurrence.instance_id] = builder
        return _EmissionTicket(occurrence.instance_id)

    def complete(self, ticket: _EmissionTicket, occurrence: Any, *, result: Any = None,
                 error: BaseException | None = None) -> None:
        if ticket.event_id != occurrence.instance_id or ticket.event_id not in self._pending:
            raise PipelineContractError("Code completion does not match a pending emission")
        builder = self._pending.pop(ticket.event_id)
        if error is not None:
            self._failed = True
        else:
            self.rank_tracker.complete(occurrence, result)
            self.events.append(builder.event())
        self._hook("complete", occurrence, result=result, error=error)

    def activate(self, owner, scope_path, lane, parameters):
        self.rank_tracker.activate(owner, scope_path, lane, parameters)
        return tuple(scope_path), self._hook("activate", owner, scope_path, lane, parameters)

    def deactivate(self, token):
        scope, provider_token = token
        self.rank_tracker.deactivate(scope)
        self.occurrence_tracker.deactivate(scope)
        self._hook("deactivate", provider_token)

    def bind_call(self, site, scope_path, lane, function, args, kwargs, source_bindings):
        rank_token = self.rank_tracker.enter_call(site, scope_path, function, args, kwargs)
        self.occurrence_tracker.enter_call(site, scope_path)
        return rank_token, self._hook("bind_call", site, scope_path, lane, function, args, kwargs, source_bindings)

    def end_call(self, token, *, result=None, error=None):
        rank_token, provider_token = token
        self.rank_tracker.leave_call(rank_token, error=error)
        self.occurrence_tracker.leave_call()
        self._hook("end_call", provider_token, result=result, error=error)

    def _builder(self, occurrence):
        path = self.occurrence_tracker.path(occurrence.site_id, occurrence.inputs.get("scope_path", ()))
        return _EventBuilder(occurrence, path)

    def finish(self) -> tuple[Event, ...]:
        if self._block is not None or self._pending or self._failed:
            raise PipelineContractError("Code window is failed or has unclosed construction state")
        events = tuple(sorted(self.events, key=lambda event: backing_value(event).occurrences[0].lane_order))
        if tuple(backing_value(event).occurrences[0].lane_order for event in events) != tuple(range(len(events))):
            raise PipelineContractError("Code window event orders must be contiguous and unique")
        if len({event.event_id for event in events}) != len(events):
            raise PipelineContractError("Code window event identities must be unique")
        return events


def _invocation_metadata(invocation: Any) -> Mapping[str, Any]:
    """Copy portable argument evidence without retaining opaque arguments."""
    values, unresolved = {}, []
    for label, bindings in (("arguments", invocation.arguments), ("globals", invocation.globals)):
        if not isinstance(bindings, Mapping) or any(type(key) is not str for key in bindings):
            raise PipelineContractError("Code invocation requires string-keyed argument/global bindings")
        result = {}
        for name, value in bindings.items():
            try:
                result[name] = safe_input(value)
            except PipelineContractError:
                unresolved.append(f"{label}.{name}")
        values[label] = result
    return {"lane": invocation.lane, **values, "unmaterialized_inputs": tuple(sorted(unresolved))}


def code_value_completion(complete, sink):
    """Pin completed facts in the actual consuming event, without extra blocks."""
    def finish(point, missing):
        value = complete(point, missing)
        sink.record_ground({"point_id": point.point_id, "query": safe_input(missing.query),
                            "inputs": safe_input(missing.inputs), "value": safe_input(value)})
        return value
    return finish


def managed_code_completion(context):
    """Evaluator-owned control-point recognition; Core resolves only Ground."""
    def complete(point, missing):
        resolution = context.ground(NeedGround("ground:" + point.point_id, point.point_id,
                                              (point.lane,), missing.query, missing.inputs))
        return context.state.get(resolution.ground_ref).result.payload
    return complete


def build_code_window(program: DynamicCodeProgram, invocations: Sequence[WorkloadInvocation], *,
                      emission_contract: Any, input_provider_factory: Any = None,
                      module_bindings_factory: Any = None, region_filter: Any = None,
                      interface_observer_factory: Any = None,
                      callback_context_provider_factory: Any = None,
                      missing_value_factory: Any = None, complete_value: Any = None,
                      region_observer_factory: Any = None, materialize_value_factory: Any = None,
                      taint_policy: CodeTaintPolicy | None = None, share: bool = True) -> EventCapture:
    """Fully run the selected mutable region, then return one closed capture.

    Runtime is an optional package, imported only by this executable builder.
    Factories receive a lane identity. An optional interface observer brackets
    the original call without executing it or supplying a replacement result.
    Callback providers carry execution context, never reusable backend state.
    A region observer brackets the selected body, borrowing entry locals only
    during its factory call. Value materialization follows portable Ground fact
    pinning; its live result is not stored as a Ground fact.
    Nothing is written to Store until the
    caller explicitly admits the returned capture through Core. Exactly one
    actual region activation per lane forms this window; ``region_filter``
    selects it without skipping execution of the other source activations.
    ``share=False`` returns validated original occurrences for worker transport;
    the coordinator must call ``share_code_capture`` after merging its ranks.
    """
    from flexeva_workload_runtime.code_window import run_code_window_workload

    if type(share) is not bool:
        raise PipelineContractError("Code sharing choice requires a Boolean")
    identity = checked_contract_identity(emission_contract.code_contract_identity)
    if not callable(getattr(emission_contract, "resolve_emission", None)):
        raise PipelineContractError("Code emission contract requires resolve_emission")
    if not isinstance(program, DynamicCodeProgram):
        raise PipelineContractError("Code window builder requires a frozen DynamicCodeProgram")
    invocations = tuple(invocations)
    lanes = tuple(invocation.lane for invocation in invocations)
    if (not lanes or any(type(lane) not in (int, str) for lane in lanes)
            or len(set(lanes)) != len(lanes)):
        raise PipelineContractError("Code window requires nonempty unique lane invocations")
    if any(invocation.mode != "instrumented" for invocation in invocations):
        raise PipelineContractError("Code window requires instrumented invocations")
    envelope = dynamic_program_envelope(program)
    frozen = restore_dynamic_program(envelope)
    policy = taint_policy if taint_policy is not None else CodeTaintPolicy()
    if not isinstance(policy, CodeTaintPolicy):
        raise PipelineContractError("Code construction requires CodeTaintPolicy")
    analysis_identity = checked_analysis_identity(policy.analysis_identity)
    analysis = CodeControlFlow(frozen, policy.sources)
    events, invocation_evidence = [], []
    for invocation in sorted(invocations, key=lambda item: (type(item.lane).__name__, str(item.lane))):
        metadata = _invocation_metadata(invocation)
        provider = input_provider_factory(invocation.lane) if input_provider_factory is not None else None
        sink = CodeWindowCollector(input_provider=provider, program=frozen, analysis=analysis, taint_policy=policy)
        modules = module_bindings_factory(invocation.lane) if module_bindings_factory is not None else None
        observer = (interface_observer_factory(invocation.lane)
                    if interface_observer_factory is not None else None)
        callback_provider = (callback_context_provider_factory(invocation.lane)
                             if callback_context_provider_factory is not None else None)
        result = run_code_window_workload(
            envelope, invocation, sink=sink, resolve_emission=emission_contract.resolve_emission,
            module_bindings=modules, region_filter=region_filter,
            interface_observer=observer, callback_context_provider=callback_provider,
            missing_value=missing_value_factory(invocation.lane) if missing_value_factory else None,
            complete_value=code_value_completion(complete_value, sink) if complete_value else None,
            region_observer=region_observer_factory(invocation.lane) if region_observer_factory else None,
            materialize_value=materialize_value_factory(invocation.lane) if materialize_value_factory else None,
            summarize_body=getattr(emission_contract, "summarize_body", None),
        )
        if (result.audit.get("protocol_version") != "code-window-v2"
                or result.audit.get("closed") is not True or result.audit.get("measured") is not True):
            raise PipelineContractError("runtime did not certify a closed instrumented Code window")
        selected = safe_input(result.audit.get("selected_region"))
        if (result.audit.get("region_count") != 1 or type(selected) is not dict
                or set(selected) != {"region_id", "activation", "occurrence"}
                or selected["region_id"] != frozen.region.region_id
                or type(selected["occurrence"]) is not int or selected["occurrence"] < 0):
            raise PipelineContractError("Code window requires exactly one selected region activation per lane")
        metadata["selected_region"] = selected
        lane_events = sink.finish()
        if result.audit.get("event_count") != len(lane_events):
            raise PipelineContractError("runtime and collector disagree on the closed Code event count")
        events.extend(lane_events)
        invocation_evidence.append(metadata)
        del result, sink, provider, modules, observer, callback_provider
    if checked_contract_identity(emission_contract.code_contract_identity) != identity:
        raise PipelineContractError("Code emission contract identity changed during construction")
    if checked_analysis_identity(policy.analysis_identity) != analysis_identity:
        raise PipelineContractError("Code taint policy identity changed during construction")
    evidence = CodeWindowEvidence(frozen, identity, tuple(invocation_evidence), analysis_identity, analysis.digest)
    root = EventCapture(frozen.source.digest, tuple(events), evidence)
    if not share:
        CodeWindowLayer(identity, analysis_identity=analysis_identity)._validate_window(root, root.events)
        return root
    from .sharing import share_code_capture
    return share_code_capture(root)


class CodeWindowLayer:
    """Four intra-layer operations over a complete evaluator-defined Code window."""

    layer_id = "code"

    def __init__(self, code_contract_identity: Mapping[str, Any], *, analysis_identity=None) -> None:
        self._code_contract_identity = checked_contract_identity(code_contract_identity)
        self._analysis_identity = checked_analysis_identity(
            CodeTaintPolicy().analysis_identity if analysis_identity is None else analysis_identity)
        self._cfg_validation = None

    @property
    def code_contract_identity(self) -> Mapping[str, Any]:
        # Core caches pipeline identity when registering. Neither the caller's
        # original mapping nor a property result can change that configuration.
        return checked_contract_identity(self._code_contract_identity)

    @property
    def pipeline_identity(self) -> Mapping[str, Any]:
        return {"adapter": "code-window-v7", "rank_analysis": checked_analysis_identity(self._analysis_identity),
                "code_contract_identity": self.code_contract_identity,
                "partition_policy": "singleton-packets-v6"}

    def _validate_capture(self, capture: EventCapture) -> None:
        from ..payloads import _payload_cache
        evidence = capture.evidence
        if not isinstance(evidence, CodeWindowEvidence) or evidence.protocol_version != "code-window-v7":
            raise PipelineContractError("CodeWindowLayer requires closed code-window-v7 evidence")
        if checked_analysis_identity(evidence.analysis_identity) != self._analysis_identity:
            raise PipelineContractError("Code analysis identity differs from the registered pipeline")
        if checked_contract_identity(evidence.code_contract_identity) != self._code_contract_identity:
            raise PipelineContractError("Code window contract identity differs from the registered pipeline")
        if not isinstance(evidence.program, DynamicCodeProgram):
            raise PipelineContractError("Code window requires its frozen dynamic program")
        if capture.revision != evidence.program.source.digest:
            raise PipelineContractError("Code window revision must identify its frozen source snapshot")
        cache = _payload_cache.get()
        key = ("code-capture", id(self), id(evidence.program), capture.revision, evidence.cfg_digest)
        if cache is not None and key in cache:
            return
        program_key = evidence.program.program_digest
        if self._cfg_validation is None or self._cfg_validation[0] != program_key:
            self._cfg_validation = (program_key, CodeControlFlow(
                evidence.program, self._analysis_identity["sources"]).digest)
        if evidence.cfg_digest != self._cfg_validation[1]:
            raise PipelineContractError("Code CFG digest differs from the frozen source analysis")
        if cache is not None:
            cache[key] = True

    def _validate_window(self, capture: EventCapture, events: Sequence[Event], context=None) -> None:
        """Check tuple projections against Code's own actual execution evidence."""
        self._validate_capture(capture)
        payload_context = context
        if getattr(context, "root", None) is not None:
            if context.root is not capture:
                self._validate_capture(context.root)
            # The program was checked for this batch; still check every event.
            payload_context = replace(context, root=None)
        from ..lanes import validate_sequence
        from .sharing import code_occurrences, iter_code_occurrences, merged_inputs
        validate_sequence(events, self.layer_id)
        # Read each immutable event once per provider in this validation only.
        # Object identity keeps distinct descriptors separate even before their
        # occurrence identities have been checked. No payload survives the call.
        payloads = {id(event): self.event_payload(event, payload_context) for event in events}
        # Occurrences are native Code data. Debug construction must not dispatch
        # per-event Payload calls through the Core's handle-only service.
        occurrence_context = payload_context
        actual = tuple(iter_code_occurrences(events, occurrence_context))
        remap = {occ.event_id: event.event_id for event, occ in actual}
        body_sources = {item["owner"]: item["source_digest"] for item in capture.evidence.program.owned_definitions}
        for event in events:
            raw = payloads[id(event)].value
            occurrences = code_occurrences(event, occurrence_context)
            for occurrence in occurrences:
                summary = occurrence.inputs.semantic_inputs.get("$body_summary")
                if summary is not None and (event.kind != "emission" or type(summary) is not dict
                            or set(summary) != {"contract", "sources"}
                            or type(summary["contract"]) is not dict or not summary["contract"]
                            or type(summary["sources"]) is not tuple or not summary["sources"]
                            or any(type(pair) is not tuple or len(pair) != 2
                                   or any(type(value) is not str for value in pair)
                                   or body_sources.get(pair[0]) != pair[1] for pair in summary["sources"])
                            or summary["sources"] != tuple(sorted(set(summary["sources"])))):
                    raise PipelineContractError("body summary source evidence differs from frozen definitions")
            if (tuple(occ.lane for occ in occurrences) != event.lane_context.active_lanes
                    or any(type(occ.lane) not in (int, str) or type(occ.lane_order) is not int
                           for occ in occurrences)):
                raise PipelineContractError("shared Code lanes disagree with actual occurrences")
            if stable_digest(raw.inputs) != stable_digest(merged_inputs(occurrences, remap, event.event_id)):
                raise PipelineContractError("shared Code inputs disagree with actual occurrences")
            for digest in (raw.structure_digest, raw.rank_fingerprint):
                if (type(digest) is not str or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)):
                    raise PipelineContractError("Code sharing requires explicit structure and rank digests")
            if (not isinstance(raw.activation_path, tuple)
                    or any(type(item) is not str for item in raw.activation_path)):
                raise PipelineContractError("Code sharing requires explicit activation and rank evidence")
        evidence = capture.evidence
        invocations = safe_input(evidence.invocations)
        lanes = tuple(item.get("lane") for item in invocations if type(item) is dict)
        if (len(lanes) != len(invocations) or not lanes
                or any(type(lane) not in (int, str) for lane in lanes)
                or len(set(lanes)) != len(lanes)):
            raise PipelineContractError("Code window evidence requires unique invocation lanes")
        known = {
            site.site_id: CodeSourceFragment(
                site.site_id, site.owner, site.span, site.source_text, site.normalized,
                site.callee, site.role, site.defs, site.uses,
            ) for site in evidence.program.sites
        }
        local_orders = {lane: 0 for lane in lanes}
        selected_regions = {}
        for item in invocations:
            selected = item.get("selected_region")
            if (type(selected) is not dict or set(selected) != {"region_id", "activation", "occurrence"}
                    or selected["region_id"] != evidence.program.region.region_id
                    or type(selected["occurrence"]) is not int or selected["occurrence"] < 0):
                raise PipelineContractError("Code window requires one selected region in each invocation")
            selected_regions[item["lane"]] = selected
        for event, occurrence in actual:
            raw = payloads[id(event)].value
            lane_id = occurrence.lane
            if lane_id not in local_orders:
                raise PipelineContractError("Code event lane is absent from the captured invocations")
            if occurrence.lane_order != local_orders[lane_id]:
                raise PipelineContractError("Code event tuple disagrees with actual per-lane execution order")
            local_orders[lane_id] += 1
            if (raw.region_id != evidence.program.region.region_id
                    or event.scope_id != raw.region_id
                    or type(raw.region_occurrence) is not int or raw.region_occurrence < 0):
                raise PipelineContractError("Code event does not belong to the captured mutable region")
            safe_input(raw.region_activation)
            if {"region_id": raw.region_id, "activation": safe_input(raw.region_activation),
                    "occurrence": raw.region_occurrence} != selected_regions[lane_id]:
                raise PipelineContractError("Code event belongs to a different selected region activation")
            if (not raw.fragments or type(raw.execution_count) is not int
                    or raw.execution_count != sum(fragment.count for fragment in raw.fragments)
                    or len({fragment.site_id for fragment in raw.fragments}) != len(raw.fragments)):
                raise PipelineContractError("Code execution summary requires counted unique source fragments")
            for fragment in raw.fragments:
                if (not isinstance(fragment, CodeSourceFragment) or type(fragment.count) is not int
                        or fragment.count < 1 or replace(fragment, count=1) != known.get(fragment.site_id)):
                    raise PipelineContractError("Code fragment differs from its frozen source site")
            if (type(occurrence.execution_digest) is not str or len(occurrence.execution_digest) != 64
                    or any(character not in "0123456789abcdef" for character in occurrence.execution_digest)):
                raise PipelineContractError("Code execution summary requires its streaming digest")
            occurrence.inputs.checked()
            binding = raw.emission_binding
            if event.kind == "emission":
                if (raw.execution_count != 1 or type(binding) is not dict
                        or set(binding) != {"transform_id", "operation_id"}
                        or any(type(value) is not str or not value for value in binding.values())):
                    raise PipelineContractError("each Code emission requires one bound source interface")
            elif binding is not None:
                raise PipelineContractError("Code block cannot carry an emission binding")

    @staticmethod
    def _raw(event: Event, context=None) -> CodeWindowEventData:
        raw = backing_value(event, context)
        if event.kind not in ("block", "emission") or not isinstance(raw, CodeWindowEventData):
            raise PipelineContractError("CodeWindowLayer received a legacy or non-Code-window event")
        return raw

    def lane(self, event: Event, context: Any) -> LaneContext:
        raw = self._payload_value(event, context)
        # Keep deterministic-content validation, without serializing/hashing a
        # fingerprint that Lane never uses.
        _canonical(self._payload_key(raw))
        lane = event.lane_context
        if lane is None or not lane.active_lanes:
            raise PipelineContractError("actual Code event requires resolved active lanes")
        return lane

    def _payload_value(self, event: Event, context: Any) -> CodeWindowEventData:
        if getattr(context, "root", None) is not None:
            self._validate_capture(context.root)
        raw = self._raw(event, context)
        return replace(raw, context=backing_context(event, context))

    def event_payload(self, event: Event, context: Any) -> EventPayload:
        raw = self._payload_value(event, context)
        key = stable_digest(self._payload_key(raw))
        return EventPayload(raw, key)

    @staticmethod
    def _payload_key(raw):
        return (raw.fragments, raw.structure_digest, raw.rank_fingerprint,
                raw.emission_binding, raw.inputs)




    def dependency_edges(self, events, context):
        """Code-specific analysis, also used by composed evaluator adapters."""
        order = {event.event_id: index for index, event in enumerate(events)}
        if len(order) != len(events):
            raise PipelineContractError("Code window has duplicate event identities")
        if context.root is not None:
            self._validate_capture(context.root)
            context = replace(context, root=None)
        edges: dict[tuple[str, str], set[str]] = {}
        for event in events:
            raw = self._payload_value(event, context)
            _canonical(self._payload_key(raw))
            for producer, symbol in raw.inputs.checked().producer_inputs:
                if producer not in order:
                    raise PipelineContractError("Code dependency names an unpublished internal or missing producer")
                if producer == event.event_id:
                    raise PipelineContractError("Code dependency must name an external producer event")
                edges.setdefault((producer, event.event_id), set()).add(symbol)
        return tuple(EventDependency(producer, consumer, symbols=tuple(sorted(symbols)))
                     for (producer, consumer), symbols in sorted(edges.items()))
