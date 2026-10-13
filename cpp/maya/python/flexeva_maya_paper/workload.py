"""GPT registration and candidates using the automatic source frontend."""

from dataclasses import asdict
import hashlib
from types import ModuleType

from flexeva_code import encode, portable
from .config import GptConfig, configuration, group_manifest
from .logical_dist import LogicalDist
from .frontend import CallRule, MayaRules, PythonWorkload, WorkloadRegistration, operand

ARTIFACT_PROVENANCE = {
    "repository": "https://github.com/SpaceVector/flexeva-ae",
    "commit": "77a9a4123316144bdae9556f61f955dbce85412a",
    "workload": "script/e3/workload/megatron/maya_megatron.py",
    "workload_sha256": "064948909ece7b2187a18d6b8bf98977c883df4893a107fda77081cbed1331e3",
    "experiment": "script/e3/figure7/measure_figure7_production.py",
    "experiment_sha256": "89b2d1fe17ebf1fb119b9021efb8a43b4aba7b8df5582de15465e1b88f7eb814",
}


def optimizer_source(candidate):
    return ("def make_optimizer(torch, parameters, learning_rate):\n"
            f"    return torch.optim.AdamW(parameters, lr=learning_rate, foreach={candidate not in ('optimizer_patch', 'P3')}, fused=False)\n")


class GptWorkload:
    def __init__(self, config="smoke", candidate="P0", *, device="cuda:0",
                 execution_mode="representative"):
        if isinstance(config, str):
            config = configuration(config, candidate)
        if type(config) is not GptConfig:
            raise TypeError("config must name a preset or be a GptConfig")
        if execution_mode not in ("representative", "full"):
            raise ValueError("unknown GPT execution mode")
        self.config, self.candidate, self.device = config, candidate, device
        self.execution_mode = execution_mode
        self._description = None
        self.last_execution = None

    def describe(self):
        if self._description is not None:
            return self._description
        from .adapter import build_plans, freeze_sources, source_hashes, _stage_inputs
        self._sources = freeze_sources(optimizer_source(self.candidate))
        # Controlled P1-P3 are explicit source edits of the frozen operator
        # function, rather than externally supplied region invalidations.
        if self.candidate in ("P1", "P2", "P3"):
            text = self._sources["_flexeva_gpt/model.py"]
            text = text.replace("F.gelu(self.up(tensor))", "F.gelu(self.up(tensor), approximate='tanh')")
            if self.candidate in ("P2", "P3"):
                text = text.replace("F.softmax(attention, dim=-1)",
                    "F.softmax(attention, dim=-1, dtype=torch.float32).to(attention.dtype)")
            self._sources["_flexeva_gpt/model.py"] = text
        groups = group_manifest(self.config)
        self.source_plan, self.rank_plan, projections = build_plans(self.config, self._sources, groups,
            device=self.device, share=self.execution_mode == "representative")
        rule = CallRule("logical-nccl-v1",
            lambda target: type(getattr(target.callable, "__self__", None)) is LogicalDist,
            lambda target: {"method": target.callable.__name__,
                            "receiver": target.callable.__self__.frontend_inputs(),
                            "args": operand(target.args), "kwargs": operand(dict(target.kwargs))})
        complete = all(group.reusable for group in self.rank_plan.groups)
        self.registration = WorkloadRegistration(MayaRules((rule,), autograd_geometry=complete),
            modeled_function="_flexeva_gpt.model.run_step",
            complete_invocation=complete)
        settings = asdict(self.config)
        self.frontend = PythonWorkload(self._sources, "_flexeva_gpt.program.run_rank",
            arguments=lambda rank: {"settings": settings, "rank": rank, "device": self.device},
            ranks=range(self.config.world_size), registration=self.registration,
            communicators=groups, rank_projections=projections)
        self.frontend.source_plan, self.frontend.rank_plan = self.source_plan, self.rank_plan
        result = self.frontend.describe()
        if complete:
            # The reviewed dense program's local kernels and schedule depend
            # on effective microbatch geometry. DP membership affects NCCL's
            # logical projection and cost; division by DP keeps its launch
            # geometry. Crossing DP=1 changes the branch and is a body change.
            stages = []
            for rank in self.rank_plan.representatives:
                geometry = _stage_inputs(self.config, rank, self.device, optimizer_source(self.candidate))
                geometry["data_parallel"] = self.config.dp > 1
                stages.append((rank, geometry))
            result["window_semantic"] = encode({"sources": self._sources,
                "policy": self.source_plan.policy_identity,
                "representative_bodies": stages, "seed": self.config.seed,
                "rules": self.registration.rules.code_contract_identity})
        hashes = source_hashes(self._sources)
        result.update({"candidate_id": self.candidate, "artifact_provenance": ARTIFACT_PROVENANCE,
            "config": self.config.as_dict(), "source_hashes": hashes,
            "source_sha256": hashlib.sha256(encode(hashes)).hexdigest(),
            "optimizer_source": optimizer_source(self.candidate),
            "controlled_invalidations": [], "mutation_kind": "source-patch",
            "frontend": {"kind": "python-ast-cfg-runtime", "manual_regions": False},
            "source_plan": {"source_digest": self.source_plan.source_digest,
                            "cfg_digest": self.source_plan.cfg_digest,
                            "policy": portable(self.source_plan.policy_identity)},
            "rank_execution_plan": self.rank_plan.as_dict()})
        self._description = result
        return result

    def execute(self, sink):
        import torch
        self.describe()
        if self.device == "cpu" and self.config.name != "smoke":
            raise ValueError("CPU execution is a smoke control check")
        communication = LogicalDist(self._description["communicators"], sink, cpu=self.device == "cpu")
        bridge = ModuleType("_flexeva_gpt_bridge")
        bridge.communication = communication
        self.frontend.module_bindings = lambda rank: {bridge.__name__: bridge}
        result = self.frontend.execute(sink)
        counts = {name: sum(output[name] for output in result["outputs"].values())
                  for name in ("model_initializations", "optimizer_initializations", "step_executions")}
        self.last_execution = {**counts, "logical_ranks": self.config.world_size,
            "actual_executed_ranks": list(result["summary"].executed_ranks),
            "representative_ranks": list(self.rank_plan.representatives),
            "code_capture": asdict(result["summary"]), "code_events_published": True,
            "validation_only": self.device == "cpu", "communication_calls": communication.calls,
            "fallback_reasons": sorted({reason for group in self.rank_plan.groups for reason in group.fallback_reason})}
        return self.last_execution
