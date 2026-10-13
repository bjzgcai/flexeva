"""Streaming actual occurrences and explicit logical-rank representation.

The collector owns only open calls, per-site counters and streaming digests.
Published packets belong to the caller's C++ RAS writer. There is no Trace list
and no invented execution digest for a logical rank that did not run.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from .evidence import CodeContractError, digest, encode, freeze, portable


@dataclass(frozen=True)
class ActualOccurrence:
    representative: int
    sequence: int
    site: str
    occurrence: tuple
    execution_digest: str


@dataclass(frozen=True)
class RepresentationEvidence:
    active_ranks: tuple[int, ...]
    mappings: object
    equivalence: object
    reusable: bool


@dataclass(frozen=True)
class CodeEvent:
    identity: str
    kind: str
    inputs: object
    actual: ActualOccurrence
    representation: RepresentationEvidence
    schema: str = "native-code-occurrence-v1"

    def as_dict(self):
        return portable({"schema": self.schema, "identity": self.identity,
            "kind": self.kind, "inputs": self.inputs,
            "actual": {"representative": self.actual.representative,
                "sequence": self.actual.sequence, "site": self.actual.site,
                "occurrence": self.actual.occurrence,
                "execution_digest": self.actual.execution_digest},
            "representation": {"active_ranks": self.representation.active_ranks,
                "mappings": self.representation.mappings,
                "equivalence": self.representation.equivalence,
                "reusable": self.representation.reusable}})


@dataclass(frozen=True)
class CaptureSummary:
    physical_occurrences: int
    logical_occurrences: int
    executed_ranks: tuple[int, ...]
    logical_ranks: tuple[int, ...]
    execution_digest: str


@dataclass(frozen=True)
class _Ticket:
    representative: int
    sequence: int
    site: str
    occurrence: tuple
    kind: str
    evidence: object


class CaptureCollector:
    """Observe the same calls that the backend uses for native capture.

    begin/end do not invoke workload code or choose returned values. publish is
    synchronous and must transfer each packet into the current RAS operation.
    The collector is unusable after a failure or finish, including a failure
    caught by workload code. A fresh candidate requires a fresh collector.
    """

    def __init__(self, execution_plan, publish):
        if not callable(publish):
            raise CodeContractError("Code capture needs a synchronous RAS packet writer")
        self.plan, self.publish = execution_plan, publish
        self._active_rank = None
        self._completed = set()
        self._stack = []
        self._site_counts = Counter()
        self._sequence = Counter()
        self._physical = self._logical = 0
        self._digest = hashlib.sha256()
        # The plan manifest stores each complete certificate once. Occurrence
        # packets refer to it rather than copying/re-hashing source closures.
        self._proof_ids = {group.representative: digest(group.proof) for group in execution_plan.groups}
        self._class_ids = {group.representative: digest(group.class_identity) for group in execution_plan.groups}
        self._digest.update(encode(execution_plan.as_dict()))
        self._closed = self._failed = False

    def _check(self):
        if self._closed or self._failed:
            raise CodeContractError("Code capture is already closed or failed")

    def begin_rank(self, representative):
        self._check()
        self.plan.group_for(representative)
        if self._active_rank is not None or representative in self._completed:
            raise CodeContractError("a representative may execute exactly once per capture")
        self._active_rank = representative

    def end_rank(self, representative):
        self._check()
        if self._active_rank != representative or self._stack:
            self.abort()
            raise CodeContractError("rank closed with mismatched or pending Code occurrences")
        self._completed.add(representative)
        self._active_rank = None

    def begin(self, site, *, evidence, kind="emission", occurrence=None):
        self._check()
        if self._active_rank is None or type(site) is not str or not site:
            raise CodeContractError("Code occurrence requires an active representative and source site")
        if kind not in ("block", "emission"):
            raise CodeContractError("Code kind must be block or emission")
        rank = self._active_rank
        order = self._sequence[rank]
        self._sequence[rank] += 1
        index = self._site_counts[rank, site]
        self._site_counts[rank, site] += 1
        # Runtime CFG hooks should pass their real activation/loop path. The
        # scalar counter is for atomic evaluator-owned repeated region scopes.
        position = freeze(occurrence if occurrence is not None else ("site-visit", index))
        if type(position) is not tuple:
            raise CodeContractError("Code occurrence position must be a tuple/list")
        ticket = _Ticket(rank, order, site, position, kind, freeze(evidence))
        self._stack.append(ticket)
        return ticket

    def end(self, ticket, *, result_evidence=None, error=None):
        self._check()
        if not self._stack or self._stack[-1] is not ticket:
            self.abort()
            raise CodeContractError("Code scopes must close in actual call order")
        self._stack.pop()
        if error is not None:
            self.abort()
            return None
        group = self.plan.group_for(ticket.representative)
        inputs = freeze({"entry": ticket.evidence, "result": result_evidence})
        actual = ActualOccurrence(ticket.representative, ticket.sequence, ticket.site,
                                  ticket.occurrence, digest(inputs))
        proof_id = self._proof_ids[ticket.representative]
        represented = RepresentationEvidence(group.active_ranks, group.mappings, proof_id,
                                              group.reusable)
        # Occurrence identity is independent of changing source/input evidence.
        # Membership and the whole-invocation certificate do not rename all
        # local occurrences when one operator changes.
        key = digest((ticket.site, ticket.occurrence, ticket.kind,
                      self._class_ids[ticket.representative]))
        event = CodeEvent(key, ticket.kind, inputs, actual, represented)
        try:
            self.publish(event)
        except BaseException:
            self.abort()
            raise
        self._physical += 1
        self._logical += len(group.active_ranks)
        self._digest.update(encode((key, actual.representative, actual.sequence,
                                   actual.execution_digest)))
        return event

    def abort(self):
        self._failed = True
        self._stack.clear()

    def finish(self):
        self._check()
        if self._active_rank is not None or self._stack or self._completed != set(self.plan.representatives):
            self.abort()
            raise CodeContractError("Code capture is missing a closed representative invocation")
        self._closed = True
        return CaptureSummary(self._physical, self._logical, tuple(sorted(self._completed)),
                              self.plan.logical_ranks, self._digest.hexdigest())
