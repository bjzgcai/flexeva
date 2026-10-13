"""Dense GPT rank rule composed with the automatic Python Code frontend."""

import hashlib
from functools import lru_cache
from importlib import metadata, util
from pathlib import Path
import sys

from flexeva_code import CodeContractError, RankEquivalenceEvidence, analyze_source, plan_rank_execution

POLICY = {"name": "maya-dense-gpt-geometry", "version": 2}
ADMITTED_SOURCES = {'_flexeva_gpt/model.py': '809ee1f4d99aa77c5549cb69471bb352c91c767dbe7d8b3e5946963345caab89', '_flexeva_gpt/program.py': '95d06a56b01ca8f6409fb7168df45f55d3bda0193e9b9dba43a2cbe9739fbdbc'}

@lru_cache(maxsize=1)
def runtime_environment():
    """Identify opaque operator rules without importing or executing PyTorch."""
    def tree_identity(package):
        spec = util.find_spec(package)
        if spec is None or spec.origin is None:
            raise CodeContractError(f"required Code frontend package is unavailable: {package}")
        directory = Path(spec.origin).parent
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(b"\x00")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    return {"python": {"implementation": sys.implementation.name,
                       "version": list(sys.version_info[:3])},
            "torch_distribution_version": metadata.version("torch"),
            "native_code_frontend_sha256": tree_identity("flexeva_code"),
            "python_code_analysis_sha256": tree_identity("flexeva.code"),
            "maya_frontend_sha256": hashlib.sha256(Path(__file__).with_name("frontend.py").read_bytes()).hexdigest(),
            "configuration_sha256": hashlib.sha256(Path(__file__).with_name("config.py").read_bytes()).hexdigest(),
            "logical_dist_sha256": hashlib.sha256(Path(__file__).with_name("logical_dist.py").read_bytes()).hexdigest(),
            "dense_rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}



def freeze_sources(candidate_optimizer):
    directory = Path(__file__).parent
    sources = {"_flexeva_gpt/" + name: (directory / name).read_text()
               for name in ("model.py", "program.py")}
    sources["_flexeva_gpt/optimizer.py"] = candidate_optimizer
    return sources


def source_hashes(sources):
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}


def _stage_inputs(config, rank, device, optimizer):
    _, _, stage = config.coordinates(rank)
    return {"stage": stage, "pipeline_stages": config.pp,
            "layer_count": config.layer_count(stage), "tensor_parallel": config.tp,
            "data_parallel": config.dp, "micro_batches": config.micro_batches,
            "micro_batch_size": config.micro_batch_size, "sequence": config.seq_len,
            "hidden": config.hidden_size, "heads": config.num_heads,
            "vocabulary": config.vocab_size, "dtype": config.dtype,
            "learning_rate": config.learning_rate, "device": device,
            "optimizer": optimizer, "schedule": "1f1b"}


def build_plans(config, sources, communicators, *, device, share):
    from .config import GptConfig
    if type(config) is not GptConfig:
        raise CodeContractError("dense GPT rank rule requires the reviewed config type")
    policy = {**POLICY, "runtime": runtime_environment()}
    plan = analyze_source(sources, "_flexeva_gpt.program.run_rank",
                          region_id="maya-gpt-rank-invocation", policy_identity=policy)
    hashes = source_hashes(sources)
    unknown = tuple("unreviewed representative-rank source: " + name
                    for name, expected in ADMITTED_SOURCES.items() if hashes.get(name) != expected)
    if not ADMITTED_SOURCES:
        unknown += ("representative-rank source rule is unavailable",)
    optimizers = {"def make_optimizer(torch, parameters, learning_rate):\n"
                  f"    return torch.optim.AdamW(parameters, lr=learning_rate, foreach={flag}, fused=False)\n"
                  for flag in (True, False)}
    optimizer = sources.get("_flexeva_gpt/optimizer.py")
    if optimizer not in optimizers:
        unknown += ("unreviewed optimizer rank rule",)
    definitions = tuple(plan.definitions)
    obligations = plan.obligations_for(definitions)
    def evidence(rank):
        _, _, stage = config.coordinates(rank)
        groups = {group["kind"]: {"id": group["id"], "local_rank": group["members"].index(rank),
                                  "members": tuple(group["members"])}
                  for group in communicators if rank in group["members"]}
        return RankEquivalenceEvidence(definitions, _stage_inputs(config, rank, device, optimizer),
            {"stage": stage, "dense_control": "shape-stage-only"}, {"rank": rank, "communicators": groups},
            complete=not unknown, unresolved=unknown,
            covered_obligations=obligations if not unknown else (), class_identity={"pipeline_stage": stage})
    ranks = plan_rank_execution(plan, range(config.world_size), evidence,
                                policy_identity=policy, share=share)
    projections = []
    for group in ranks.groups:
        original = group.mappings[str(group.representative)]["communicators"]
        for rank in group.active_ranks:
            target = group.mappings[str(rank)]["communicators"]
            projections.append({"rank": rank, "representative": group.representative,
                                "communicator_map": [(original[k]["id"], target[k]["id"])
                                                     for k in ("tp", "pp", "dp")]})
    return plan, ranks, sorted(projections, key=lambda row: row["rank"])
