"""Pre-setup execution groups certified by a versioned evaluator rule.

CFG taint is supporting evidence, never authorization to skip calls. Rules must
cover source-visible controls, opaque effects, complete performance inputs and
logical resource mappings for the whole invocation being skipped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .evidence import CodeContractError, digest, freeze, identity, portable


@dataclass(frozen=True)
class RankEquivalenceEvidence:
    source_definitions: tuple[str, ...]
    semantic_inputs: object
    control_signature: object
    mapping: object
    complete: bool = False
    unresolved: tuple[str, ...] = ()
    covered_obligations: tuple[str, ...] = ()
    ground_refs: tuple = ()
    # Rule-defined logical class coordinate, stable across content changes.
    # It must distinguish classes with different behavior in this invocation;
    # membership, source revision and concrete input values are not its identity.
    class_identity: object = None


@dataclass(frozen=True)
class RankGroup:
    representative: int
    active_ranks: tuple[int, ...]
    mappings: object
    proof: object
    reusable: bool
    fallback_reason: tuple[str, ...] = ()
    class_identity: object = None


@dataclass(frozen=True)
class RankExecutionPlan:
    groups: tuple[RankGroup, ...]
    logical_ranks: tuple[int, ...]
    source_digest: str
    policy_identity: object
    schema: str = "native-rank-plan-v1"

    @property
    def representatives(self):
        return tuple(group.representative for group in self.groups)

    @property
    def fallback_ranks(self):
        return tuple(group.representative for group in self.groups if group.fallback_reason)

    def group_for(self, representative):
        for group in self.groups:
            if group.representative == representative:
                return group
        raise CodeContractError(f"rank {representative} is not an execution representative")

    def as_dict(self):
        return portable({"schema": self.schema, "source_digest": self.source_digest,
                         "policy_identity": self.policy_identity, "logical_ranks": self.logical_ranks,
                         "groups": tuple({"representative": group.representative,
                             "active_ranks": group.active_ranks, "mappings": group.mappings,
                             "proof": group.proof, "reusable": group.reusable,
                             "fallback_reason": group.fallback_reason,
                             "class_identity": group.class_identity} for group in self.groups)})


def plan_rank_execution(source, ranks, evidence_provider, *, policy_identity, share=True):
    """Group before model construction, or explicitly execute uncertain ranks.

    The provider must not execute/initialize a workload. It supplies evaluator
    knowledge beyond the source analysis, including a complete effects contract.
    A missing/partial certificate is a visible full-execution fallback, never a
    singleton equivalent-class claim and never permission to cache its results.
    Ground failures propagate; they cannot be repaired by all-rank no-op calls.
    """
    policy = identity(policy_identity, "rank equivalence")
    ranks = tuple(ranks)
    if (not ranks or any(type(rank) is not int or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)):
        raise CodeContractError("rank plan requires distinct nonnegative logical ranks")
    if type(share) is not bool:
        raise CodeContractError("rank sharing must be Boolean")
    buckets, mappings, certificates, reasons, classes = defaultdict(list), {}, {}, {}, {}
    class_proofs = {}
    for rank in sorted(ranks):
        item = evidence_provider(rank)
        if not isinstance(item, RankEquivalenceEvidence):
            raise CodeContractError("rank rule must return RankEquivalenceEvidence")
        if type(item.complete) is not bool:
            raise CodeContractError("rank input/effects completeness must be explicit")
        required = set(source.obligations_for(item.source_definitions))
        missing = tuple(sorted(required - set(item.covered_obligations)))
        why = tuple(item.unresolved) + missing
        if source.entrypoint not in source.closure(item.source_definitions):
            why += ("source certificate does not cover the invocation entrypoint",)
        if not item.complete:
            why += ("incomplete invocation inputs/effects certificate",)
        if item.class_identity is None:
            why += ("missing stable execution class identity",)
        proof = freeze({"schema": "native-rank-equivalence-v1", "policy": policy,
                        "inputs": source.evidence_for(item.source_definitions, item.semantic_inputs,
                                                       ground_refs=item.ground_refs),
                        "control": item.control_signature,
                        "class_identity": item.class_identity,
                        "covered_obligations": tuple(sorted(set(item.covered_obligations)))})
        proof_id = digest(proof)
        stable_class = freeze({"policy": policy, "class": item.class_identity})
        if not why:
            class_key = digest(stable_class)
            previous = class_proofs.setdefault(class_key, proof_id)
            if previous != proof_id:
                raise CodeContractError("execution class identity aliases different invocation evidence")
        key = proof_id if share and not why else f"rank:{rank}"
        buckets[key].append(rank)
        mappings[rank] = freeze(item.mapping)
        certificates[key] = proof
        reasons[key] = tuple(sorted(set(why)))
        classes[key] = stable_class if not why else freeze({"fallback_rank": rank, "policy": policy})
    groups = tuple(RankGroup(members[0], tuple(members),
                            freeze({str(rank): mappings[rank] for rank in members}),
                            certificates[key], not bool(reasons[key]), reasons[key], classes[key])
                   for key, members in sorted(buckets.items(), key=lambda pair: pair[1][0]))
    return RankExecutionPlan(groups, tuple(sorted(ranks)), source.source_digest, policy)
