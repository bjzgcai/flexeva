"""Code-owned cross-rank event construction, before RAS partitioning."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace

from ..construction import ordered_groups
from ..errors import PipelineContractError
from ..lanes import lane_key
from ..model import Event, LaneContext, PayloadBinding, stable_digest
from ..payloads import backing_value
from .inputs import DynamicInputEvidence
from .window import CodeOccurrence, CodeWindowEventData, CodeWindowLayer


def code_occurrences(event, context=None):
    """Read explicit execution evidence; singleton events use the same schema."""
    raw = backing_value(event, context)
    if (not isinstance(raw, CodeWindowEventData) or not isinstance(raw.occurrences, tuple)
            or not raw.occurrences or any(not isinstance(occ, CodeOccurrence) for occ in raw.occurrences)):
        raise PipelineContractError("Code events require explicit nonempty actual occurrences")
    return raw.occurrences


def iter_code_occurrences(events, context=None):
    """Yield (source event, actual occurrence), without manufacturing RAS events."""
    seen = set()
    for event in events:
        for occurrence in code_occurrences(event, context):
            if type(occurrence.event_id) is not str or not occurrence.event_id:
                raise PipelineContractError("Code occurrences require original event identities")
            if occurrence.event_id in seen:
                raise PipelineContractError("Code occurrences must have unique original identities")
            seen.add(occurrence.event_id)
            yield event, occurrence


def merged_inputs(occurrences, remap, event_id):
    return DynamicInputEvidence(
        producer_inputs=tuple(sorted({(remap.get(producer, producer), symbol)
            for occ in occurrences for producer, symbol in occ.inputs.producer_inputs
            if remap.get(producer, producer) != event_id})),
        semantic_inputs={str(lane_key(occ.lane)): occ.inputs.semantic_inputs for occ in occurrences},
        source_inputs={str(lane_key(occ.lane)): occ.inputs.source_inputs for occ in occurrences},
        state_refs=tuple(dict.fromkeys(ref for occ in occurrences for ref in occ.inputs.state_refs)),
        inputs_complete=all(occ.inputs.inputs_complete for occ in occurrences),
        unresolved=tuple(sorted({item for occ in occurrences for item in occ.inputs.unresolved})),
    ).checked()


@dataclass(frozen=True)
class _Member:
    """Construction-local lane evidence, not a second Code event format."""

    event_id: str
    lane_context: LaneContext
    source: Event
    occurrence: CodeOccurrence


def share_code_capture(capture):
    adapter = CodeWindowLayer(capture.evidence.code_contract_identity,
                              analysis_identity=capture.evidence.analysis_identity)
    adapter._validate_window(capture, capture.events)
    # Reuse isolated reads only within this construction, including when a
    # shared input has several actual occurrences. Keep both window checks.
    bodies = {id(event): backing_value(event) for event in capture.events}
    context = SimpleNamespace(payload=lambda event: SimpleNamespace(value=bodies[id(event)]))
    bases = {}
    for event in capture.events:
        raw = bodies[id(event)]
        identity = (raw.region_id, raw.region_activation, raw.region_occurrence,
                    raw.activation_path, event.kind,
                    tuple((fragment.site_id, fragment.count) for fragment in raw.fragments),
                    raw.structure_digest, raw.emission_binding)
        bases[id(event)] = stable_digest(identity)
    counts, keys, members = Counter(), [], []
    for event, occurrence in iter_code_occurrences(capture.events, context):
        raw, base, lane = bodies[id(event)], bases[id(event)], occurrence.lane
        index = counts[lane, base]
        counts[lane, base] += 1
        keys.append(stable_digest((base, index, raw.rank_fingerprint)))
        members.append(_Member(occurrence.event_id,
            LaneContext((lane,)), event, occurrence))
    groups = ordered_groups(tuple(members), keys)
    remap = {member.event_id: "code-shared:" + key[:32] for key, group in groups for member in group}
    shared = []
    for _, group in groups:
        first = group[0]
        event_id = remap[first.event_id]
        occurrences = tuple(member.occurrence for member in group)
        lanes = tuple(occ.lane for occ in occurrences)
        inputs = merged_inputs(occurrences, remap, event_id)
        shared.append(replace(first.source, event_id=event_id, match_key=event_id,
            lane_context=LaneContext(lanes),
            # A source event can contribute to multiple output groups. Keep
            # their mutable metadata independent, as well as input isolation.
            binding=PayloadBinding.inline(replace(deepcopy(bodies[id(first.source)]), occurrences=occurrences, inputs=inputs)), payload_fingerprint=None))
    result = replace(capture, events=tuple(shared))
    adapter._validate_window(result, result.events)
    return result
