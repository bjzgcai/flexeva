"""Execute frozen Python once while publishing Code to an external RAS sink.

Native interface observers and Ground completion run in this same execution.
This module neither runs a downstream transform nor owns any native Trace.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import hashlib

from .capture import ActualOccurrence, RepresentationEvidence, CodeEvent, CaptureSummary
from .evidence import CodeContractError, digest, encode, freeze


@lru_cache(maxsize=32)
def _instrument(files, region_id, entrypoint):
    from flexeva.model import SourceFile, SourceSnapshot
    from flexeva.code.dynamic import export_dynamic_region
    from flexeva.code.region_model import MutableRegion
    source = SourceSnapshot.create(tuple(SourceFile.from_text(path, text) for path, text in files))
    return export_dynamic_region(source, MutableRegion(region_id, tuple(path for path, _ in files), entrypoint))


def capture_instrumented(source, execution_plan, invocation_factory, emission_contract, publish,
                         *, input_provider_factory=None, module_bindings_factory=None,
                         interface_observer_factory=None, missing_value_factory=None,
                         complete_value=None, materialize_value_factory=None,
                         callback_context_provider_factory=None, taint_policy=None,
                         state_ref_encoder=None, lifecycle_factory=None,
                         region_observer_factory=None, result_observer=None,
                         execution_context_factory=None):
    """Run one actual invocation per representative and stream Code packets.

    invocation_factory receives the representative. Factories must not execute
    another dry run. The optional observer brackets the very same Python calls
    that produce native records. Ground's complete_value must pin its result in
    the active root-capture operation before returning it to the workload.
    Exceptions propagate, including missing-value completion without a handler;
    the caller must abort its RAS transaction on any failure.
    """
    from flexeva.code.cfg import CodeControlFlow
    from flexeva.code.dynamic import dynamic_program_envelope
    from flexeva.code.rank import RankTracker
    from flexeva.code.taint import CodeTaintPolicy
    from flexeva.code.window import CodeWindowCollector, CodeWindowEventData, code_value_completion
    from flexeva.model import BackendState, PayloadBinding
    from flexeva_workload_runtime.code_window import run_code_window_workload

    if source.source_digest != execution_plan.source_digest:
        raise CodeContractError("execution plan belongs to another frozen source")
    if any(group.proof["inputs"]["analysis_policy"] != source.policy_identity
           for group in execution_plan.groups):
        raise CodeContractError("execution plan belongs to another source analysis policy")
    if not callable(publish):
        raise CodeContractError("instrumented capture requires a RAS packet writer")
    files = tuple((item.path, item.text) for item in source.program.source.files)
    program = _instrument(files, source.region_id, source.entrypoint)
    envelope = dynamic_program_envelope(program)
    policy = taint_policy if taint_policy is not None else CodeTaintPolicy()
    analysis = CodeControlFlow(program, policy.sources)
    stream = hashlib.sha256(encode(execution_plan.as_dict()))
    total = logical = 0
    executed = []

    for group in execution_plan.groups:
        rank = group.representative
        count = 0
        proof_id = digest(group.proof)
        class_id = digest(group.class_identity)
        site_positions = {}
        identities = {}
        locations = {}
        lifecycle = lifecycle_factory(rank) if lifecycle_factory else None

        def emit(event):
            nonlocal count, total, logical
            # This is the collector's freshly materialized, checked inline
            # packet. Borrow it only for this synchronous callback, then detach
            # into our immutable CodeEvent. The public backing_value accessor
            # still copies arbitrary published payloads for its callers.
            binding = event.binding
            if (not isinstance(binding, PayloadBinding)
                    or not isinstance(binding.state, BackendState)
                    or not isinstance(binding.state.value, CodeWindowEventData)):
                raise CodeContractError("streaming collector requires its owned inline Code payload")
            raw = binding.state.value
            if len(raw.occurrences) != 1 or raw.occurrences[0].lane != rank:
                raise CodeContractError("runtime emitted a fabricated or foreign actual occurrence")
            old = raw.occurrences[0]
            refs = old.inputs.state_refs
            if refs and state_ref_encoder is None:
                raise CodeContractError("managed inputs require an explicit native RAS reference encoder")
            producers = []
            for producer, symbol in old.inputs.producer_inputs:
                if producer not in identities:
                    raise CodeContractError("Code dependency names an unreserved producer")
                producers.append((identities[producer][0], symbol))
            inputs = freeze({"producer_inputs": tuple(producers),
                "semantic_inputs": old.inputs.semantic_inputs, "source_inputs": old.inputs.source_inputs,
                "state_refs": tuple(state_ref_encoder(ref) for ref in refs),
                "inputs_complete": old.inputs.inputs_complete,
                "unresolved": old.inputs.unresolved,
                "fragments": tuple((fragment.site_id, fragment.count) for fragment in raw.fragments),
                "source_location": locations[event.event_id]})
            key, site, position = identities[event.event_id]
            actual = ActualOccurrence(rank, old.lane_order, site, freeze(position), old.execution_digest)
            representation = RepresentationEvidence(group.active_ranks, group.mappings, proof_id,
                                                      old.inputs.inputs_complete)
            packet = CodeEvent(key, event.kind, inputs, actual, representation)
            publish(packet)
            stream.update(encode((key, rank, old.lane_order, old.execution_digest)))
            count += 1
            total += 1
            logical += len(group.active_ranks)

        class Events:
            """The legacy collector's append surface, without a retained list."""
            def append(self, event):
                emit(event)

        class StreamingCollector(CodeWindowCollector):
            def _evidence(self, occurrence):
                evidence = super()._evidence(occurrence)
                if (occurrence.kind == "block" and lifecycle is not None
                        and hasattr(lifecycle, "observe_block")):
                    refs = evidence.state_refs
                    if refs and state_ref_encoder is None:
                        raise CodeContractError("managed inputs require an explicit native RAS reference encoder")
                    # Preserve actual order and every repeated value. Legacy
                    # execution_digest contains window-order producer IDs and
                    # remains diagnostic only; these coordinates are stable.
                    lifecycle.observe_block(identities[occurrence.instance_id][0], {
                        "site": occurrence.site_id,
                        "producer_inputs": tuple((identities[item][0], symbol)
                                                 for item, symbol in evidence.producer_inputs),
                        "semantic_inputs": evidence.semantic_inputs,
                        "source_inputs": evidence.source_inputs,
                        "state_refs": tuple(state_ref_encoder(ref) for ref in refs),
                        "inputs_complete": evidence.inputs_complete,
                        "unresolved": evidence.unresolved})
                return evidence

            def activate(self, owner, scope_path, lane, parameters):
                token = super().activate(owner, scope_path, lane, parameters)
                if lifecycle is not None and hasattr(lifecycle, "activate"):
                    lifecycle.activate(owner)
                return owner, token

            def deactivate(self, token):
                owner, original = token
                try:
                    super().deactivate(original)
                finally:
                    if lifecycle is not None and hasattr(lifecycle, "deactivate"):
                        lifecycle.deactivate(owner)

            def _builder(self, occurrence):
                builder = super()._builder(occurrence)
                site = occurrence.site_id
                path = freeze((builder.region_activation, builder.region_occurrence,
                               *builder.activation_path))
                last_path, ordinal = site_positions.get(site, (None, -1))
                ordinal = ordinal + 1 if last_path == path else 0
                site_positions[site] = path, ordinal
                position = (*path, "site-visit", ordinal)
                key = digest((site, position, occurrence.kind, class_id))
                identities[occurrence.instance_id] = key, site, position
                locations[occurrence.instance_id] = freeze({
                    name: occurrence.site.get(name) for name in
                    ("owner", "callee", "role", "source_text", "span")})
                return builder

            def arrive(self, occurrence):
                ticket = super().arrive(occurrence)
                if lifecycle is not None:
                    key, _, position = identities[occurrence.instance_id]
                    builder = self._pending[ticket.event_id]
                    producer_refs = tuple((identities[item][0], symbol)
                                          for item, symbol in builder.producers)
                    lifecycle.begin(key, occurrence, freeze({
                        "producer_inputs": producer_refs,
                        "semantic_inputs": builder.semantic,
                        "source_inputs": builder.sources,
                        "inputs_complete": builder.complete,
                        "unresolved": tuple(sorted(builder.unresolved)),
                        "occurrence": position}))
                return ticket

            def complete(self, ticket, occurrence, *, result=None, error=None):
                try:
                    super().complete(ticket, occurrence, result=result, error=error)
                except BaseException as failure:
                    if lifecycle is not None:
                        lifecycle.end(identities[occurrence.instance_id][0], failure)
                    raise
                else:
                    if lifecycle is not None:
                        lifecycle.end(identities[occurrence.instance_id][0], error)

            def finish(self):
                if self._block is not None or self._pending or self._failed:
                    raise CodeContractError("instrumented Code capture failed or left open occurrences")
                return count

        def factory(callback):
            return callback(rank) if callback is not None else None

        collector = StreamingCollector(input_provider=factory(input_provider_factory), program=program,
                                       analysis=analysis, taint_policy=policy)
        collector.events = Events()
        if not group.reusable:
            # Explicit full execution does not claim rank equivalence. Keep the
            # occurrence and def-use trackers, but do not require a proof of rank
            # transfer through opaque calls that are actually executed per rank.
            collector.rank_tracker = RankTracker(None, CodeTaintPolicy(sources=()))
        invocation = invocation_factory(rank)
        if invocation.lane != rank or invocation.mode != "instrumented":
            raise CodeContractError("invocation must execute its representative in instrumented mode")
        with factory(execution_context_factory) if execution_context_factory else nullcontext():
            result = run_code_window_workload(envelope, invocation, sink=collector,
                resolve_emission=emission_contract.resolve_emission,
                module_bindings=factory(module_bindings_factory),
                interface_observer=factory(interface_observer_factory),
                callback_context_provider=factory(callback_context_provider_factory),
                missing_value=factory(missing_value_factory),
                complete_value=code_value_completion(complete_value, collector) if complete_value else None,
                materialize_value=factory(materialize_value_factory),
                region_observer=factory(region_observer_factory),
                summarize_body=getattr(emission_contract, "summarize_body", None),
                source_callable=getattr(emission_contract, "source_callable", None))
        audit = result.audit
        if (audit.get("closed") is not True or audit.get("measured") is not True
                or audit.get("region_count") != 1 or audit.get("event_count") != collector.finish()):
            raise CodeContractError("runtime did not close exactly one Code window")
        executed.append(rank)
        if result_observer is not None:
            result_observer(rank, result.output, result.audit)
    return CaptureSummary(total, logical, tuple(executed), execution_plan.logical_ranks, stream.hexdigest())
