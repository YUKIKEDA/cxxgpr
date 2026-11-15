# SPDX-License-Identifier: Apache-2.0
"""Synchronize or verify mirror copies between .agents/skills and .claude/skills."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCE_ROOT = Path(".agents/skills")
TARGET_ROOT = Path(".claude/skills")


def collect_files(root: Path) -> dict[Path, bytes]:
    """Read all regular files under root excluding Python bytecode caches."""
    if not root.exists():
        return {}
    entries: dict[Path, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        entries[rel] = path.read_bytes()
    return entries


def find_mismatches(source_root: Path, target_root: Path) -> list[str]:
    """Return human-readable mismatch descriptions between source and target mirrors."""
    source_files = collect_files(source_root)
    target_files = collect_files(target_root)
    errors: list[str] = []

    if not source_files:
        errors.append(f"No skill files found under {source_root.as_posix()}")
        return errors

    for rel, content in source_files.items():
        if rel not in target_files:
            errors.append(f"Missing mirror file: {(target_root / rel).as_posix()}")
        elif target_files[rel] != content:
            errors.append(f"Content mismatch: {(target_root / rel).as_posix()}")

    for rel in target_files:
        if rel not in source_files:
            errors.append(f"Unexpected extra mirror file: {(target_root / rel).as_posix()}")

    return errors


def sync_mirrors(source_root: Path, target_root: Path) -> int:
    """Copy all files from source_root to target_root, pruning stale entries."""
    source_files = collect_files(source_root)
    if target_root.exists():
        shutil.rmtree(target_root)
    for rel, content in source_files.items():
        dest = target_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return len(source_files)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify mirrors match without modifying files.",
    )
    args = parser.parse_args()

    if args.check:
        errors = find_mismatches(SOURCE_ROOT, TARGET_ROOT)
        for error in errors:
            print(error)
        return 1 if errors else 0

    count = sync_mirrors(SOURCE_ROOT, TARGET_ROOT)
    print(f"Synced {count} files from {SOURCE_ROOT.as_posix()} to {TARGET_ROOT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
