"""Construct caller-declared sharing from an already ordered event tuple.

Per-lane constraints are projections of the input tuple, not separately supplied
ordinals. Callers own equivalence; Core's refresh does not invoke this builder.
"""

import heapq
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from .errors import PipelineContractError
from .lanes import lane_key, validate_lane
from .model import stable_digest


def ordered_groups(events: Sequence, keys: Sequence[str]):
    if not isinstance(events, tuple):
        raise PipelineContractError("sharing construction requires an ordered event tuple")
    if len(events) != len(keys):
        raise ValueError("each construction event requires an equivalence key")
    groups = defaultdict(list)
    lanes = defaultdict(list)
    for event, key in zip(events, keys):
        validate_lane(event.lane_context)
        groups[key].append(event)
        for lane in event.lane_context.active_lanes:
            lanes[lane].append(key)
    # A shared event represents at most one occurrence on each lane.
    invalid = {key for key, members in groups.items()
               if len({lane for event in members for lane in event.lane_context.active_lanes})
               != sum(len(event.lane_context.active_lanes) for event in members)}
    successors = {key: set() for key in groups}
    incoming = {key: 0 for key in groups}
    for items in lanes.values():
        for before, after in pairwise(items):
            if before == after:
                invalid.add(before)
            elif after not in successors[before]:
                successors[before].add(after)
                incoming[after] += 1
    # Preserve input interleaving whenever sharing does not require a change.
    priority = {key: index for index, key in enumerate(groups)}
    ready = [(priority[key], key) for key in groups if incoming[key] == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, key = heapq.heappop(ready)
        ordered.append(key)
        for after in sorted(successors[key]):
            incoming[after] -= 1
            if incoming[after] == 0:
                heapq.heappush(ready, (priority[after], after))
    invalid.update(set(groups) - set(ordered))
    if invalid:
        if all(len(groups[key]) == 1 for key in invalid):
            raise PipelineContractError("input lane sequence cannot be linearized")
        # Split conflicting equivalence classes, preserving all actual events.
        split = [stable_digest((key, event.event_id)) if key in invalid and len(groups[key]) > 1 else key
                 for event, key in zip(events, keys)]
        if split == list(keys):
            raise PipelineContractError("input lane sequence cannot be linearized")
        return ordered_groups(events, split)
    return tuple((key, tuple(sorted(groups[key], key=lambda event:
        tuple(lane_key(lane) for lane in event.lane_context.active_lanes)))) for key in ordered)
