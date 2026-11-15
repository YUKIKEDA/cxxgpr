# SPDX-License-Identifier: Apache-2.0
"""Check or accept hexfloat numerical baselines under qa/baselines/."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import cast

BASELINES_ROOT = Path("qa/baselines")
SCHEMA = "cxxgpr.baseline/1"


def _parse_canary_output(stdout: str) -> tuple[str, dict[str, str]]:
    platform: str | None = None
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {line_no}: expected 2 tokens, got {line!r}")
        key, val = parts
        if key == "platform" and platform is None:
            platform = val
            continue
        try:
            parsed = float.fromhex(val)
        except ValueError as exc:
            raise ValueError(f"Line {line_no}: invalid hexfloat for {key!r}: {val!r}") from exc
        if math.isnan(parsed):
            raise ValueError(f"Line {line_no}: NaN value for {key!r} cannot form a baseline")
        values[key] = float.hex(parsed)

    if not platform:
        raise ValueError("Canary output did not report a 'platform <key>' header line")
    if not values:
        raise ValueError("Canary output produced zero metric values")
    return platform, values


def _run_canary(binary: Path) -> tuple[str, dict[str, str]]:
    if not binary.exists():
        raise FileNotFoundError(f"Canary executable not found: {binary.as_posix()}")
    proc = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return _parse_canary_output(proc.stdout)


def _baseline_path(case_name: str, platform: str) -> Path:
    return BASELINES_ROOT / case_name / f"{platform}.json"


def _ieee754_bits(hex_str: str) -> bytes:
    return struct.pack("!d", float.fromhex(hex_str))


def cmd_accept(case_name: str, binary: Path) -> int:
    """Capture canary output and store it as the certified baseline for this platform."""
    platform, values = _run_canary(binary)
    target = _baseline_path(case_name, platform)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "case": case_name,
        "platform": platform,
        "values": values,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Accepted {len(values)} metrics for {platform} -> {target.as_posix()}")
    return 0


def cmd_check(case_name: str, binary: Path, *, if_certified: bool) -> int:
    """Compare live canary output bit for bit against the certified platform baseline."""
    platform, actual = _run_canary(binary)
    target = _baseline_path(case_name, platform)
    if not target.exists():
        msg = f"Uncertified platform '{platform}' (missing {target.as_posix()})"
        if if_certified:
            print(f"NOTICE: {msg}; skipping bitwise comparison (--if-certified).")
            return 0
        print(f"ERROR: {msg}. See docs/quality/result-changes/README.md.")
        return 1

    raw_doc: object = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw_doc, dict):
        print(f"ERROR: malformed baseline JSON at {target.as_posix()}")
        return 1
    doc = cast(dict[str, object], raw_doc)
    if (
        doc.get("schema") != SCHEMA
        or doc.get("case") != case_name
        or doc.get("platform") != platform
    ):
        print(f"ERROR: baseline metadata mismatch in {target.as_posix()}")
        return 1

    raw_expected = doc.get("values")
    if not isinstance(raw_expected, dict):
        print(f"ERROR: missing values map in {target.as_posix()}")
        return 1
    expected = {str(k): str(v) for k, v in cast(dict[object, object], raw_expected).items()}

    diffs: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            diffs.append(f"+ {key}: {actual[key]} (not in baseline)")
        elif key not in actual:
            diffs.append(f"- {key}: {expected[key]} (missing from run)")
        elif _ieee754_bits(actual[key]) != _ieee754_bits(expected[key]):
            diffs.append(f"! {key}: expected {expected[key]}, got {actual[key]}")

    if diffs:
        print(f"Bitwise regression detected for '{platform}' against {target.as_posix()}:")
        for diff in diffs:
            print(f"  {diff}")
        return 1

    print(f"Bitwise canary check passed for '{platform}' ({len(actual)} metrics matched).")
    return 0


def cmd_platforms(case_name: str) -> int:
    """List certified platform baselines for a case."""
    case_dir = BASELINES_ROOT / case_name
    if not case_dir.exists():
        print("(none)")
        return 0
    platforms = sorted(p.stem for p in case_dir.glob("*.json"))
    for platform in platforms:
        print(platform)
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    accept_p = sub.add_parser("accept", help="Accept current canary output as platform baseline.")
    accept_p.add_argument("--case", required=True)
    accept_p.add_argument("--run", required=True, type=Path)

    check_p = sub.add_parser("check", help="Check canary output bit-for-bit against baseline.")
    check_p.add_argument("--case", required=True)
    check_p.add_argument("--run", required=True, type=Path)
    check_p.add_argument(
        "--if-certified",
        action="store_true",
        help="Do not fail when no certified baseline JSON exists for the current runner platform.",
    )

    plat_p = sub.add_parser("platforms", help="List certified platforms for a case.")
    plat_p.add_argument("--case", required=True)

    args = parser.parse_args()
    if args.cmd == "accept":
        return cmd_accept(str(args.case), cast(Path, args.run))
    if args.cmd == "check":
        return cmd_check(
            str(args.case),
            cast(Path, args.run),
            if_certified=bool(args.if_certified),
        )
    return cmd_platforms(str(args.case))


if __name__ == "__main__":
    raise SystemExit(main())
