# SPDX-License-Identifier: Apache-2.0
"""Generate and verify requirement/design/format/QA traceability against tests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

DESIGN_DOC = Path("docs/design.md")
PERSIST_DOC = Path("docs/persist-format.md")
QUALITY_DOC = Path("docs/quality/quality-plan.md")
CHANGES_DOC = Path("docs/quality/result-changes/README.md")
MATRIX_DOC = Path("docs/quality/traceability.md")
TESTS_DIR = Path("tests")
VERIFICATION_DIR = Path("docs/verification")

SPEC_ID_RE = re.compile(r"\b((?:REQ|DES|FMT|QA)-\d+)\b")
REQ_ROW_RE = re.compile(r"^\|\s*(REQ-\d+)\s*\|\s*([^|]+?)\s*\|")
HEADING_SPEC_RE = re.compile(r"^#{2,3}\s+((?:DES|FMT|QA)-\d+)\s+(.*)$")
TEST_MACRO_RE = re.compile(r"^\s*TEST(?:_F|_P)?\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)")
SPEC_COMMENT_RE = re.compile(r"//\s*Spec:\s*(.+)$")
SPEC_PROPERTY_RE = re.compile(r'RecordProperty\s*\(\s*"spec"\s*,\s*"([^"]+)"\s*\)')


@dataclass(frozen=True)
class SpecItem:
    """Single canonical specification identifier and its document location."""

    spec_id: str
    title: str
    source: str
    line: int


def _empty_list() -> list[str]:
    return []


@dataclass
class Coverage:
    """Tests and verification records mapped to a specification identifier."""

    tests: list[str] = field(default_factory=_empty_list)
    verifications: list[str] = field(default_factory=_empty_list)


def _strip_code_fences(lines: list[str]) -> list[tuple[int, str]]:
    in_fence = False
    out: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append((idx, line))
    return out


def parse_definitions() -> tuple[dict[str, SpecItem], list[str]]:
    """Parse canonical spec IDs from design, persist-format, and quality docs."""
    specs: dict[str, SpecItem] = {}
    errors: list[str] = []

    def add_spec(spec_id: str, title: str, source: Path, line_no: int) -> None:
        if spec_id in specs:
            prev = specs[spec_id]
            errors.append(
                f"Duplicate spec ID {spec_id} at {source.as_posix()}:{line_no} "
                f"(previously defined at {prev.source}:{prev.line})"
            )
            return
        specs[spec_id] = SpecItem(
            spec_id=spec_id,
            title=title.strip(),
            source=source.as_posix(),
            line=line_no,
        )

    for doc in (DESIGN_DOC, PERSIST_DOC, QUALITY_DOC):
        if not doc.exists():
            errors.append(f"Missing canonical spec doc: {doc.as_posix()}")
            continue
        lines = doc.read_text(encoding="utf-8").splitlines()
        for line_no, line in _strip_code_fences(lines):
            if doc == DESIGN_DOC and (req_match := REQ_ROW_RE.match(line)):
                add_spec(req_match.group(1), req_match.group(2), doc, line_no)
            elif head_match := HEADING_SPEC_RE.match(line):
                add_spec(head_match.group(1), head_match.group(2), doc, line_no)

    return specs, errors


def _extract_ids(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]


def collect_links(defined_ids: set[str]) -> tuple[dict[str, Coverage], list[str]]:
    """Collect test and verification coverage links and report unknown ID references."""
    coverage: dict[str, Coverage] = {spec_id: Coverage() for spec_id in defined_ids}
    errors: list[str] = []

    def record_test_ref(spec_id: str, label: str, loc: str) -> None:
        if spec_id not in defined_ids:
            errors.append(f"{loc}: reference to undefined spec ID '{spec_id}'")
            return
        cov = coverage[spec_id]
        if label not in cov.tests:
            cov.tests.append(label)

    if TESTS_DIR.exists():
        for cpp_file in sorted(TESTS_DIR.rglob("*.cpp")):
            rel = cpp_file.relative_to(Path()).as_posix()
            lines = cpp_file.read_text(encoding="utf-8").splitlines()
            pending_comment_ids: list[tuple[int, str]] = []
            current_test: str | None = None
            file_level_ids: list[tuple[int, str]] = []

            for line_no, line in enumerate(lines, start=1):
                if comment_match := SPEC_COMMENT_RE.search(line):
                    tokens = _extract_ids(comment_match.group(1))
                    for token in tokens:
                        if current_test is not None:
                            record_test_ref(token, f"{rel} ({current_test})", f"{rel}:{line_no}")
                        else:
                            pending_comment_ids.append((line_no, token))
                    continue

                if prop_match := SPEC_PROPERTY_RE.search(line):
                    label = f"{rel} ({current_test})" if current_test else rel
                    for token in _extract_ids(prop_match.group(1)):
                        record_test_ref(token, label, f"{rel}:{line_no}")

                if line.strip().startswith("namespace") and pending_comment_ids:
                    file_level_ids.extend(pending_comment_ids)
                    pending_comment_ids.clear()

                if test_match := TEST_MACRO_RE.match(line):
                    current_test = f"{test_match.group(1)}.{test_match.group(2)}"
                    for comment_line, token in pending_comment_ids:
                        record_test_ref(token, f"{rel} ({current_test})", f"{rel}:{comment_line}")
                    pending_comment_ids.clear()
                    continue

                if pending_comment_ids and line.strip() and not line.strip().startswith("//"):
                    file_level_ids.extend(pending_comment_ids)
                    pending_comment_ids.clear()

            for comment_line, token in [*file_level_ids, *pending_comment_ids]:
                record_test_ref(token, rel, f"{rel}:{comment_line}")

    if VERIFICATION_DIR.exists():
        for json_path in sorted(VERIFICATION_DIR.rglob("*.json")):
            rel = json_path.as_posix()
            try:
                raw: object = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{rel}: invalid JSON ({exc})")
                continue
            if not isinstance(raw, dict):
                continue
            raw_map = cast(dict[str, object], raw)
            raw_specs = raw_map.get("specs")
            if isinstance(raw_specs, list):
                for item in cast(list[object], raw_specs):
                    if not isinstance(item, str):
                        continue
                    if item not in defined_ids:
                        errors.append(f"{rel}: undefined spec ID '{item}'")
                    elif rel not in coverage[item].verifications:
                        coverage[item].verifications.append(rel)

    for doc in (DESIGN_DOC, PERSIST_DOC, QUALITY_DOC, CHANGES_DOC):
        if not doc.exists():
            continue
        lines = doc.read_text(encoding="utf-8").splitlines()
        for line_no, line in _strip_code_fences(lines):
            for ref in SPEC_ID_RE.findall(line):
                if ref not in defined_ids:
                    errors.append(f"{doc.as_posix()}:{line_no}: undefined spec reference '{ref}'")

    return coverage, errors


def _spec_sort_key(spec_id: str) -> tuple[int, int]:
    prefix, num = spec_id.split("-", 1)
    order = {"REQ": 0, "DES": 1, "FMT": 2, "QA": 3}
    return (order.get(prefix, 99), int(num))


def render_matrix(specs: dict[str, SpecItem], coverage: dict[str, Coverage]) -> str:
    """Render deterministic Markdown content for docs/quality/traceability.md."""
    lines: list[str] = [
        "# 仕様追跡表",
        "",
        "このファイルは `tools/qa/trace.py matrix` によって自動生成される。手編集しない。",
        "",
        "| ID | 項目 | 正本位置 | テスト | 検証記録 |",
        "|---|---|---|---|---|",
    ]
    for spec_id in sorted(specs, key=_spec_sort_key):
        spec = specs[spec_id]
        cov = coverage[spec_id]
        rel_doc = (
            spec.source.removeprefix("docs/quality/")
            if spec.source.startswith("docs/quality/")
            else f"../{spec.source.removeprefix('docs/')}"
        )
        loc_link = f"[{Path(spec.source).name}#L{spec.line}]({rel_doc}#L{spec.line})"
        tests_cell = "<br>".join(f"`{t}`" for t in cov.tests) if cov.tests else "—"
        verif_cell = "<br>".join(f"`{v}`" for v in cov.verifications) if cov.verifications else "—"
        safe_title = spec.title.replace("|", "\\|")
        lines.append(f"| `{spec_id}` | {safe_title} | {loc_link} | {tests_cell} | {verif_cell} |")
    lines.append("")
    return "\n".join(lines)


def run_check() -> int:
    """Verify definitions, references, and traceability.md freshness."""
    specs, def_errors = parse_definitions()
    coverage, ref_errors = collect_links(set(specs))
    errors = [*def_errors, *ref_errors]

    expected = render_matrix(specs, coverage)
    if not MATRIX_DOC.exists():
        errors.append(f"Missing generated traceability matrix: {MATRIX_DOC.as_posix()}")
    else:
        actual = MATRIX_DOC.read_text(encoding="utf-8")
        if actual != expected:
            errors.append(
                f"{MATRIX_DOC.as_posix()} is out of date; "
                "run 'python -X utf8 tools/qa/trace.py matrix'"
            )

    for error in errors:
        print(error)
    if not errors:
        print(f"Traceability check passed ({len(specs)} specifications tracked).")
    return 1 if errors else 0


def run_matrix() -> int:
    """Regenerate docs/quality/traceability.md."""
    specs, def_errors = parse_definitions()
    coverage, ref_errors = collect_links(set(specs))
    errors = [*def_errors, *ref_errors]
    for error in errors:
        print(error)
    if errors:
        return 1
    content = render_matrix(specs, coverage)
    MATRIX_DOC.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_DOC.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {MATRIX_DOC.as_posix()} ({len(specs)} specifications).")
    return 0


def run_show(spec_id: str) -> int:
    """Print summary and mapped tests/verifications for one specification ID."""
    specs, def_errors = parse_definitions()
    coverage, ref_errors = collect_links(set(specs))
    for error in [*def_errors, *ref_errors]:
        print(error)
    if spec_id not in specs:
        print(f"Unknown spec ID: {spec_id}")
        return 1
    spec = specs[spec_id]
    cov = coverage[spec_id]
    print(f"{spec.spec_id}: {spec.title} ({spec.source}:{spec.line})")
    print(f"Tests: {cov.tests or ['(none)']}")
    print(f"Verifications: {cov.verifications or ['(none)']}")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Check spec references and traceability.md freshness.")
    sub.add_parser("matrix", help="Regenerate docs/quality/traceability.md.")
    show_p = sub.add_parser("show", help="Show details for one spec ID.")
    show_p.add_argument("spec_id")
    args = parser.parse_args()

    if args.cmd == "check":
        return run_check()
    if args.cmd == "matrix":
        return run_matrix()
    return run_show(str(args.spec_id))


if __name__ == "__main__":
    raise SystemExit(main())
