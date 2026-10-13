#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Python with Maya's FakeCUDA libraries bound over CUDA library paths."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--proot", default="proot")
    parser.add_argument("--library-dir", action="append", default=[], type=Path,
                        help="additional directory containing CUDA libraries")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Python arguments after --")
    args = parser.parse_args()
    if os.environ.get("LD_PRELOAD"):
        parser.error("unset LD_PRELOAD before using FakeCUDA file bindings")
    proot = shutil.which(args.proot)
    python = shutil.which(args.python)
    if not proot or not python:
        parser.error("both the selected Python interpreter and PRoot must exist")
    libraries = args.lib_dir.resolve(strict=True)
    paths = json.loads(subprocess.check_output(
        [python, "-c", "import json,sys; print(json.dumps(sys.path))"], text=True))
    search = set(args.library_dir)
    for path in paths:
        if path:
            search.update((Path(path) / "nvidia").glob("*/lib"))
            search.add(Path(path) / "torch/lib")
    search.update((Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib64")))
    search.update(Path("/usr/local").glob("cuda*/lib64"))
    directories = sorted({p.resolve() for p in search if p.is_dir()})
    mapping = {
        "libcuda.so.1": "libcuda.so*",
        "libcudart.so.12": "libcudart.so*",
        "libcublas.so.12": "libcublas.so*",
        "libcublasLt.so.12": "libcublasLt.so*",
        "libnccl.so.2": "libnccl.so*",
        "libnvidia-ml.so": "libnvidia-ml.so*",
    }
    command, bound = [proot], set()
    for name, pattern in mapping.items():
        source = libraries / name
        if not source.is_file():
            parser.error(f"missing FakeCUDA library: {source}")
        for directory in directories:
            for target in directory.glob(pattern):
                if not target.is_file():
                    continue
                for path in (target, target.resolve()):
                    if path not in bound and path.parent != libraries:
                        command.extend(("-b", f"{source}:{path}"))
                        bound.add(path)
    environment = dict(os.environ)
    library_path = [str(libraries), *(str(p) for p in directories)]
    if environment.get("LD_LIBRARY_PATH"):
        library_path.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_path)
    environment["FLEXEVA_FAKE_LIB_DIR"] = str(libraries)
    arguments = args.args[1:] if args.args[:1] == ["--"] else args.args
    with tempfile.TemporaryDirectory(prefix="flexeva-proot-") as temporary:
        environment["PROOT_TMP_DIR"] = temporary
        return subprocess.call([*command, python, *arguments], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
