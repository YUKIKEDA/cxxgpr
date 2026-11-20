# SPDX-License-Identifier: Apache-2.0
"""Check or accept hexfloat numerical baselines under qa/baselines/."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import cast

BASELINES_ROOT = Path("qa/baselines")
RESULT_CHANGES_DIR = Path("docs/internal/quality/result-changes")
RECORD_NUMBER_RE = re.compile(r"^\d{4}$")
SCHEMA = "cxxgpr.baseline/1"


def _parse_canary_output(stdout: str) -> tuple[str, bool, dict[str, str]]:
    platform: str | None = None
    uncommitted_changes: bool | None = None
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
        if key == "uncommitted_changes" and uncommitted_changes is None:
            if val == "true":
                uncommitted_changes = True
            elif val == "false":
                uncommitted_changes = False
            else:
                raise ValueError(
                    f"Line {line_no}: invalid boolean for uncommitted_changes: {val!r}"
                )
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
    if uncommitted_changes is None:
        raise ValueError(
            "Canary output did not report an 'uncommitted_changes <true|false>' header line"
        )
    if not values:
        raise ValueError("Canary output produced zero metric values")
    return platform, uncommitted_changes, values


def _run_canary(binary: Path) -> tuple[str, bool, dict[str, str]]:
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


def _baseline_path(case_name: str, platform: str, baselines_root: Path = BASELINES_ROOT) -> Path:
    return baselines_root / case_name / f"{platform}.json"


def _ieee754_bits(hex_str: str) -> bytes:
    return struct.pack("!d", float.fromhex(hex_str))


def _validate_result_change_record(case_name: str, record: str, *, changes_dir: Path) -> str | None:
    if not RECORD_NUMBER_RE.match(record):
        return f"Invalid --record '{record}' (expected 4-digit NNNN)"
    if not changes_dir.exists():
        return f"Missing result-changes directory: {changes_dir.as_posix()}"
    matches = sorted(changes_dir.glob(f"{record}-*.md"))
    if not matches:
        return f"Missing result change record '{record}-*.md' under {changes_dir.as_posix()}"
    if len(matches) > 1:
        names = [m.name for m in matches]
        return f"Ambiguous result change record '{record}-*.md': {names}"
    text = matches[0].read_text(encoding="utf-8")
    if case_name not in text:
        return f"Result change record '{matches[0].as_posix()}' does not mention case '{case_name}'"
    return None


def _accept_parsed(
    *,
    case_name: str,
    platform: str,
    uncommitted_changes: bool,
    values: dict[str, str],
    record: str | None = None,
    allow_dirty: bool = False,
    baselines_root: Path = BASELINES_ROOT,
    changes_dir: Path = RESULT_CHANGES_DIR,
) -> int:
    if uncommitted_changes and not allow_dirty:
        print(
            "ERROR: Refusing to accept baseline from a dirty git tree "
            "(uncommitted_changes=true; commit first or pass --allow-dirty for diagnostics)."
        )
        return 1

    target = _baseline_path(case_name, platform, baselines_root)
    if target.exists() or record is not None:
        if record is None:
            print(
                f"ERROR: Existing baseline {target.as_posix()} requires --record NNNN "
                f"referencing a record under {changes_dir.as_posix()}."
            )
            return 1
        err = _validate_result_change_record(case_name, record, changes_dir=changes_dir)
        if err is not None:
            print(f"ERROR: {err}")
            return 1

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


def cmd_accept(
    case_name: str,
    binary: Path,
    *,
    record: str | None = None,
    allow_dirty: bool = False,
) -> int:
    """Capture canary output and store it as the certified baseline for this platform."""
    platform, uncommitted_changes, values = _run_canary(binary)
    return _accept_parsed(
        case_name=case_name,
        platform=platform,
        uncommitted_changes=uncommitted_changes,
        values=values,
        record=record,
        allow_dirty=allow_dirty,
    )


def cmd_check(case_name: str, binary: Path, *, if_certified: bool) -> int:
    """Compare live canary output bit for bit against the certified platform baseline."""
    platform, _uncommitted, actual = _run_canary(binary)
    target = _baseline_path(case_name, platform)
    if not target.exists():
        msg = f"Uncertified platform '{platform}' (missing {target.as_posix()})"
        if if_certified:
            print(f"NOTICE: {msg}; skipping bitwise comparison (--if-certified).")
            return 0
        print(f"ERROR: {msg}. See docs/internal/quality/result-changes/README.md.")
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


def _self_test() -> int:
    try:
        _parse_canary_output("platform win-test\nx 0x1.0p+0\n")
    except ValueError:
        pass
    else:
        print("Self-test expected ValueError when uncommitted_changes header is missing")
        return 1

    platform, uncommitted, values = _parse_canary_output(
        "platform win-test\nuncommitted_changes false\nx 0x1.0p+0\n"
    )
    if platform != "win-test" or uncommitted or "x" not in values:
        print("Self-test failed to parse valid canary output")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baselines = root / "baselines"
        changes = root / "result-changes"
        changes.mkdir(parents=True)

        # 1. Dirty git tree is rejected even on initial registration.
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=True,
                values=values,
                baselines_root=baselines,
                changes_dir=changes,
            )
            == 0
        ):
            print("Self-test expected dirty tree accept to fail")
            return 1

        # 2. Clean initial baseline without existing file is permitted without --record.
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=False,
                values=values,
                baselines_root=baselines,
                changes_dir=changes,
            )
            != 0
        ):
            print("Self-test expected clean initial accept to succeed")
            return 1

        # 3. Updating existing baseline without --record is rejected.
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=False,
                values=values,
                baselines_root=baselines,
                changes_dir=changes,
            )
            == 0
        ):
            print("Self-test expected existing baseline update without --record to fail")
            return 1

        # 4. Missing record file is rejected.
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=False,
                values=values,
                record="0001",
                baselines_root=baselines,
                changes_dir=changes,
            )
            == 0
        ):
            print("Self-test expected missing record file to fail")
            return 1

        # 5. Record file not mentioning the case name is rejected.
        rec_file = changes / "0001-unrelated.md"
        rec_file.write_text("# 0001: other change\n- Case: other-case\n", encoding="utf-8")
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=False,
                values=values,
                record="0001",
                baselines_root=baselines,
                changes_dir=changes,
            )
            == 0
        ):
            print("Self-test expected record omitting case name to fail")
            return 1

        # 6. Record mentioning the case name allows update.
        rec_file.write_text(
            "# 0001: update baseline\n- Case: numeric-canary\n",
            encoding="utf-8",
        )
        updated_values = {"x": float.hex(2.0)}
        if (
            _accept_parsed(
                case_name="numeric-canary",
                platform="win-test",
                uncommitted_changes=False,
                values=updated_values,
                record="0001",
                baselines_root=baselines,
                changes_dir=changes,
            )
            != 0
        ):
            print("Self-test expected documented record update to succeed")
            return 1

    print("Results gate self-test passed.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in QA-04/QA-06 validation self-test.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    accept_p = sub.add_parser("accept", help="Accept current canary output as platform baseline.")
    accept_p.add_argument("--case", required=True)
    accept_p.add_argument("--run", required=True, type=Path)
    accept_p.add_argument(
        "--record",
        default=None,
        help=(
            "4-digit NNNN of docs/internal/quality/result-changes/NNNN-*.md (required for updates)."
        ),
    )
    accept_p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Diagnostic escape hatch allowing uncommitted_changes=true.",
    )

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
    if args.self_test:
        return _self_test()
    if args.cmd is None:
        parser.error("one of the arguments accept check platforms --self-test is required")
    if args.cmd == "accept":
        return cmd_accept(
            str(args.case),
            cast(Path, args.run),
            record=cast("str | None", args.record),
            allow_dirty=bool(args.allow_dirty),
        )
    if args.cmd == "check":
        return cmd_check(
            str(args.case),
            cast(Path, args.run),
            if_certified=bool(args.if_certified),
        )
    return cmd_platforms(str(args.case))


if __name__ == "__main__":
    raise SystemExit(main())
