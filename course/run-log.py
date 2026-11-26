"""Record UTF-8 source files, the GPU inventory and one training run in a TXT."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def record_source(source, output):
    if not source.is_dir() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Use a code-only --source directory, with the log outside it")
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Source snapshot contains a symlink: {relative}")
        if path.is_file():
            contents = path.read_bytes().decode("utf-8")
            if "\0" in contents:
                raise ValueError(f"Source snapshot contains a binary file: {relative}")
            files.append(json.dumps({"path": relative.as_posix(), "contents": contents}, ensure_ascii=False))
    if not files:
        raise ValueError("The source snapshot is empty")
    return "\n".join(files)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--header", required=True, type=Path)
    cli.add_argument("--source", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("command", nargs=argparse.REMAINDER)
    args = cli.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        cli.error("Provide the training command after --")
    if args.output.suffix != ".txt" or args.output.exists():
        cli.error("--output must name a new .txt file")

    header = args.header.read_text(encoding="utf-8").strip()
    code = record_source(args.source, args.output)
    gpu = subprocess.check_output(["nvidia-smi"], encoding="utf-8").rstrip()
    with args.output.open("x", encoding="utf-8", buffering=1) as log:
        log.write(f"{header}\nCommand: {shlex.join(command)}\n\n[nvidia-smi]\n{gpu}\n\n[Code]\n{code}\n\n[Training]\n")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding="utf-8", errors="replace") as run:
            for line in run.stdout:
                log.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            status = run.wait()
        log.write(f"\n[Run status]\nExit Code: {status}\n")
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
