# SPDX-License-Identifier: Apache-2.0
"""Release helpers: version check, archive creation and checksums.

python tools/release/package.py check-version [--tag vX.Y.Z]
python tools/release/package.py archive --stage DIR --platform NAME --out DIR
python tools/release/package.py checksums --dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md")


def cmake_version() -> str:
    """Return the project version declared in the top-level CMakeLists.txt."""
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(cxxgpr\s+VERSION\s+(\d+\.\d+\.\d+)", text)
    if match is None:
        raise SystemExit("ERROR: could not find the project version in CMakeLists.txt")
    return match.group(1)


def check_version(tag: str) -> int:
    """Fail unless the tag (when given) equals ``v`` plus the CMake project version."""
    version = cmake_version()
    if tag and tag != f"v{version}":
        print(f"ERROR: tag {tag} does not match the CMake version {version} (expected v{version}).")
        return 1
    print(f"version {version}" + (f" matches tag {tag}" if tag else " (no tag given)"))
    return 0


def _files(stage: Path) -> list[Path]:
    return sorted(p for p in stage.rglob("*") if p.is_file())


def archive(stage: Path, platform: str, out: Path) -> Path:
    """Pack the install tree and the root documents into one archive.

    The archive is a ``.zip`` for Windows platforms and a ``.tar.gz`` otherwise. The top-level
    directory is ``cxxgpr-<version>-<platform>``.
    """
    if not (stage / "include" / "cxxgpr" / "gpr" / "gpr.hpp").is_file():
        raise SystemExit(f"ERROR: {stage} does not look like an installed cxxgpr tree")
    name = f"cxxgpr-{cmake_version()}-{platform}"
    out.mkdir(parents=True, exist_ok=True)
    members = [(p, Path(name) / p.relative_to(stage)) for p in _files(stage)]
    members += [(ROOT / f, Path(name) / f) for f in ROOT_FILES if (ROOT / f).is_file()]
    if platform.startswith("windows"):
        target = out / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for src, arc in members:
                zf.write(src, arc.as_posix())
    else:
        target = out / f"{name}.tar.gz"
        with tarfile.open(target, "w:gz") as tf:
            for src, arc in members:
                tf.add(src, arc.as_posix(), recursive=False)
    print(f"wrote {target} ({len(members)} files)")
    return target


def checksums(directory: Path) -> Path:
    """Write ``SHA256SUMS.txt`` for every file in the directory."""
    lines: list[str] = []
    for path in sorted(
        p for p in directory.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    target = directory / "SHA256SUMS.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(target.read_text(encoding="utf-8"), end="")
    return target


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    p_check = sub.add_parser("check-version")
    p_check.add_argument("--tag", default="")
    p_arch = sub.add_parser("archive")
    p_arch.add_argument("--stage", type=Path, required=True)
    p_arch.add_argument("--platform", required=True)
    p_arch.add_argument("--out", type=Path, required=True)
    p_sum = sub.add_parser("checksums")
    p_sum.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "version":
        print(cmake_version())
        return 0
    if args.command == "check-version":
        return check_version(args.tag)
    if args.command == "archive":
        archive(args.stage, args.platform, args.out)
        return 0
    checksums(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
