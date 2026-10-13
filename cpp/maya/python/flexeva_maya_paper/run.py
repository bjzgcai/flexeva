"""Run the frozen GPT trajectory through the native FlexEva Core."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from .workload import GptWorkload


def _execution_evidence(workload):
    if workload.last_execution is not None:
        return {"capture_performed": True, **workload.last_execution}
    # Native no-op and projection-only refreshes intentionally never call the
    # workload. An absent callback means zero work, not missing measurement.
    return {"capture_performed": False, "logical_ranks": workload.config.world_size,
            "actual_executed_ranks": [], "representative_ranks": list(workload.rank_plan.representatives),
            "model_initializations": 0, "optimizer_initializations": 0,
            "step_executions": 0, "setup_seconds": 0.0, "step_seconds": 0.0,
            "cleanup_seconds": 0.0, "region_entries": 0, "communication_calls": 0,
            "fallback_reasons": sorted({reason for group in workload.rank_plan.groups
                                        for reason in group.fallback_reason}),
            "code_events_published": False, "validation_only": workload.device == "cpu",
            "code_capture": {"physical_occurrences": 0, "logical_occurrences": 0,
                             "executed_ranks": [], "logical_ranks": list(workload.rank_plan.logical_ranks),
                             "execution_digest": None}}


def _counters(snapshot):
    metrics = snapshot["metrics"]
    return {
        "raw_records_generated": metrics["raw_records"],
        "retained_native_calls": metrics["retained_calls"],
        "raw_bytes_written": metrics["raw_bytes"],
        "cost_calls": metrics["cost_calls"],
        "scheduled_events": metrics["scheduled_events"],
        "published_versions": metrics["published_versions"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "core": dict(metrics["core"]),
        "memory": dict(metrics["memory"]),
        **{key: value for key, value in metrics.items()
           if isinstance(value, (int, float)) and not key.endswith("_seconds")},
    }


def _build_identity(native):
    paths = {Path(native.__file__).resolve()}
    # Only the extension and loaded FakeCUDA DSOs; do not hash complete Torch
    # weight/profile directories or infer a build identity from source alone.
    for line in Path("/proc/self/maps").read_text().splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) == 6 and pieces[5].startswith("/"):
            path = Path(pieces[5])
            cuda_library = path.name.startswith(("libcuda.", "libcudart.", "libcublas.",
                                                  "libcublasLt.", "libnccl.", "libfake_cuda_core."))
            if ("fake" in str(path).lower() or cuda_library) and path.is_file() and ".so" in path.name:
                paths.add(path.resolve())
    # PRoot can expose mapped target paths in /proc/self/maps. Include the
    # actual bound source images supplied by the launcher as well.
    directory = os.environ.get("FLEXEVA_FAKE_LIB_DIR")
    if directory:
        for path in Path(directory).glob("lib*.so*"):
            if path.is_file():
                paths.add(path.resolve())
    artifacts = []
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        artifacts.append({"path": str(path), "sha256": digest.hexdigest()})
    return {"artifacts": artifacts}


def _probes(native, session, first, last, workload):
    history_before = native.inspect(session, first)["normalized_digest"]
    parent_before = native.inspect(session, last)["normalized_digest"]
    description = workload.describe()
    no_op_started = time.monotonic_ns()
    no_op = native.refresh(session, last, description, workload.execute)
    no_op_elapsed = time.monotonic_ns() - no_op_started
    counters = _counters(no_op)
    failure_description = copy.deepcopy(workload.describe())
    failure_description["window_semantic"] += b"\x00intentional-failure"
    failure_description["window_complete"] = False

    def fail(sink):
        raise RuntimeError("intentional GPT acceptance failure")

    before_memory = native.memory(session)
    observed = False
    try:
        native.refresh(session, last, failure_description, fail)
    except RuntimeError as error:
        if "intentional GPT acceptance failure" not in str(error):
            raise
        observed = True
    after_memory = native.memory(session)
    return {
        "no_op": {**counters, "elapsed_ns": no_op_elapsed,
                  "phases_ns": {key.removesuffix("_seconds"): int(value * 1e9)
                                for key, value in no_op["metrics"].items()
                                if key.endswith("_seconds")},
                  "feedback_before": last["feedback"],
                  "feedback_after": no_op["feedback"]},
        "history": {"before_digest": history_before,
                    "after_digest": native.inspect(session, first)["normalized_digest"]},
        "failure": {"before_digest": parent_before,
                    "after_digest": native.inspect(session, last)["normalized_digest"],
                    "exception_observed": observed,
                    "published_versions": after_memory["publish_count"] - before_memory["publish_count"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--mode", choices=("full-rank", "full-compact", "refresh"), default="refresh")
    parser.add_argument("--trajectory", choices=("controlled", "optimizer"), default="optimizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.chunk_bytes < 1024:
        parser.error("--chunk-bytes must be at least 1024")
    import torch  # Interpreter/library startup is outside candidate intervals.
    import _flexeva_maya_paper as native
    session = native.new_session(mode=args.mode, chunk_bytes=args.chunk_bytes)
    candidate_ids = ("P0", "P1", "P2", "P3", "P4") if args.trajectory == "controlled" else ("P0", "optimizer_patch")
    ready_ns = time.monotonic_ns()
    rows = []
    parent = first = None
    for candidate_id in candidate_ids:
        print(f"starting {args.config}/{args.mode}/{candidate_id}", flush=True)
        started = time.monotonic_ns()
        workload = GptWorkload(args.config, candidate_id,
                               execution_mode="representative" if args.mode == "refresh" else "full")
        description = workload.describe()
        prepared = time.monotonic_ns()
        if parent is None or args.mode != "refresh":
            snapshot = native.init(session, description, workload.execute)
        else:
            snapshot = native.refresh(session, parent, description, workload.execute)
        completed = time.monotonic_ns()
        # Inspection/export is diagnostic work, never an input to evaluation.
        inspection = native.inspect(session, snapshot)
        phases = {key.removesuffix("_seconds"): int(value * 1e9)
                  for key, value in snapshot["metrics"].items() if key.endswith("_seconds")}
        phases["native_total"] = phases.pop("total")
        phases["python_input_prepare"] = prepared - started
        rows.append({
            "id": candidate_id, "config": description["config"],
            "artifact_provenance": description["artifact_provenance"],
            "source_sha256": description["source_sha256"],
            "source_hashes": description["source_hashes"],
            "optimizer_source": description["optimizer_source"],
            "controlled_invalidations": description["controlled_invalidations"],
            "mutation_kind": description["mutation_kind"],
            "source_plan": description["source_plan"],
            "rank_execution_plan": description["rank_execution_plan"],
            "execution_evidence": _execution_evidence(workload),
            **inspection,
            "feedback": snapshot["feedback"],
            "timing": {"input_restore_started_ns": started,
                       "feedback_ready_ns": completed, "phases_ns": phases},
            "counters": {**_counters(snapshot), **{key: inspection[key]
                         for key in ("logical_events", "physical_records", "trace_bytes")}},
        })
        # Keep completed candidate evidence on a timeout without presenting an
        # unfinished trajectory as an accepted result. This diagnostic I/O is
        # outside the input-restoration-to-feedback interval.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output.with_suffix(".partial.json")
        checkpoint.write_text(json.dumps({"status": "incomplete", "config": args.config,
                                         "mode": args.mode, "trajectory": args.trajectory,
                                         "candidates": rows}, sort_keys=True) + "\n")
        print(f"finished {candidate_id}: {(completed - started) / 1e9:.3f}s, "
              f"raw={snapshot['metrics']['raw_records']}", flush=True)
        parent = native.commit(session, snapshot)
        if first is None:
            first = parent
    probes = _probes(native, session, first, parent, workload) if args.mode == "refresh" else {}
    build = _build_identity(native)
    result = {"schema": "flexeva.maya.paper.run.v1", "config": args.config,
              "mode": args.mode, "trajectory": args.trajectory,
              "cost_model": "live-deterministic", "torch_version": torch.__version__,
              "ready_ns": ready_ns, "finished_ns": time.monotonic_ns(),
              "build": build, "candidates": rows, "probes": probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
