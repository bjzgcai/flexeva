"""Layer-local sequence invariants, independent of reuse dependencies."""

from collections import defaultdict
from collections.abc import Sequence

from .errors import PipelineContractError
from .model import Event, EventPartition, LaneContext


def lane_key(lane: int | str) -> tuple[str, str]:
    return type(lane).__name__, str(lane)


def lane_signature(event: Event) -> tuple[int | str, ...]:
    return tuple(sorted(event.lane_context.active_lanes, key=lane_key))


def projection_partitions(events: Sequence[Event], prefix: str, *, scope=None) -> tuple[EventPartition, ...]:
    """Group equal lane sets consecutive in every active lane's projection.

    ``scope`` optionally supplies an evaluator boundary which groups must not
    cross. Disjoint events never interrupt an open group. Work is linear in
    the total number of active-lane memberships, without scanning tuple gaps.
    """
    groups, candidates, last_owner = [], {}, {}
    for event in events:
        signature = lane_signature(event)
        key = (scope(event) if scope else None, signature)
        owner = candidates.get(key)
        if owner is None or any(last_owner.get(lane) != owner for lane in signature):
            owner = len(groups)
            groups.append([])
            candidates[key] = owner
        groups[owner].append(event)
        for lane in signature:
            last_owner[lane] = owner
    return tuple(EventPartition(prefix + group[0].event_id,
                              tuple(event.event_id for event in group), events=tuple(group))
                 for group in groups)


def validate_partition(partition: EventPartition) -> None:
    """Debug audit of one newly created or changed partition's descriptors."""
    if not partition.partition_id or not partition.events:
        raise PipelineContractError("RAS partitions must be named and nonempty")
    validate_sequence(partition.events, partition.partition_id)
    ids = tuple(event.event_id for event in partition.events)
    if ids != partition.event_ids or len(set(ids)) != len(ids):
        raise PipelineContractError("partition descriptors disagree with their member index")
    signature = lane_signature(partition.events[0])
    for event in partition.events:
        if event.payload_fingerprint is None:
            raise PipelineContractError("partition returned an unmaterialized event")
        if lane_signature(event) != signature:
            raise PipelineContractError("partition events require identical active-lane sets")
    if partition.lane_signature != signature:
        raise PipelineContractError("partition lane signature disagrees with its descriptors")


def validate_partitions(events: Sequence[Event], partitions: Sequence[EventPartition], *,
                        check_contents: bool = True) -> None:
    """Check equal active sets and projection continuity independently of DataDep."""
    by_id = {event.event_id: event for event in events}
    owners = {}
    for partition in partitions:
        if not partition.event_ids:
            raise PipelineContractError("RAS partitions must be nonempty")
        signature = None
        for event_id in partition.event_ids:
            if event_id not in by_id or event_id in owners:
                raise PipelineContractError("partitions must cover every event exactly once")
            event = by_id[event_id]
            if check_contents:
                current = lane_signature(event)
                if signature is not None and current != signature:
                    raise PipelineContractError("partition events require identical active-lane sets")
                signature = current
            owners[event_id] = partition.partition_id
        if check_contents and partition.lane_signature and partition.lane_signature != signature:
            raise PipelineContractError("partition lane signature disagrees with its descriptors")
    if set(owners) != set(by_id):
        raise PipelineContractError("partitions must cover every event exactly once")
    last_owner, closed = {}, set()
    for event in events:
        owner = owners[event.event_id]
        for lane in event.lane_context.active_lanes:
            previous = last_owner.get(lane)
            if previous != owner:
                if (lane, owner) in closed:
                    raise PipelineContractError("partition is not consecutive in its lane projection")
                if previous is not None:
                    closed.add((lane, previous))
                last_owner[lane] = owner


def validate_lane(context: LaneContext) -> None:
    lanes = context.active_lanes
    if (not isinstance(lanes, tuple)
            or any(type(lane) not in (int, str) for lane in lanes)
            or len(set(lanes)) != len(lanes)):
        raise PipelineContractError("Lane requires unique int/string lanes")


def validate_sequence(events: Sequence[Event], layer_id: str) -> None:
    """Validate an ordered RAS representation, not a backend's native chronology.

    Per-lane order is its tuple projection. Producers preserve native order;
    Core preserves their tuples and detects changes through those projections.
    """
    if not isinstance(events, tuple):
        raise PipelineContractError(f"layer {layer_id!r} events must be an ordered tuple")
    for event in events:
        context = event.lane_context
        if not isinstance(context, LaneContext):
            raise PipelineContractError(f"layer {layer_id!r} has an unmaterialized Lane")
        validate_lane(context)
        if not context.active_lanes:
            raise PipelineContractError(f"layer {layer_id!r} stores an inactive event")


def projections(events: Sequence[Event]) -> dict[int | str, tuple[str, ...]]:
    lanes = defaultdict(list)
    for event in events:
        for lane in event.lane_context.active_lanes:
            lanes[lane].append(event.event_id)
    return {lane: tuple(ids) for lane, ids in lanes.items()}


def reordered_events(old: Sequence[Event], new: Sequence[Event]) -> set[str]:
    """Only shared-lane relative order matters, not disjoint-lane interleaving."""
    before, after = projections(old), projections(new)
    changed = set()
    for lane in before.keys() & after.keys():
        common = set(before[lane]) & set(after[lane])
        if (tuple(key for key in before[lane] if key in common)
                != tuple(key for key in after[lane] if key in common)):
            changed.update(common)
    return changed
