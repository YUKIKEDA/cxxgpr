# SPDX-License-Identifier: Apache-2.0
"""Verify module dependency layering and third-party header isolation."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

ARCH_DOC = Path("docs/architecture.md")
PUBLIC_ROOT = Path("include/cxxgpr")
SRC_ROOT = Path("src")
TESTS_ROOT = Path("tests")

TABLE_ROW_RE = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*([^|]+)\|\s*[^|]+\|\s*$")
BACKTICK_MOD_RE = re.compile(r"`([a-z_]+)`")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+([<"])([^>"]+)[>"]')
CXXGPR_INCLUDE_RE = re.compile(r"^cxxgpr/([a-z_]+)/")

EIGEN_MODULES = frozenset({"linalg", "kernel", "workspace", "gpr"})
OPENMP_MODULES = frozenset({"linalg", "kernel", "workspace", "gpr"})
JSON_MODULES = frozenset({"persist"})


def parse_allowed_deps(doc_path: Path) -> tuple[dict[str, set[str]], list[str]]:
    """Parse permitted direct module dependency table from docs/architecture.md."""
    if not doc_path.exists():
        return {}, [f"Missing architecture doc: {doc_path.as_posix()}"]

    deps: dict[str, set[str]] = {}
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW_RE.match(line)
        if not match:
            continue
        module = match.group(1)
        deps_cell = match.group(2).strip()
        if "なし" in deps_cell:
            deps[module] = set()
        else:
            deps[module] = set(BACKTICK_MOD_RE.findall(deps_cell))

    if not deps:
        return {}, [f"Could not parse module dependency matrix from {doc_path.as_posix()}"]
    return deps, []


def _module_for_public_header(path: Path, public_root: Path) -> str | None:
    rel = path.relative_to(public_root)
    return rel.parts[0] if len(rel.parts) >= 2 else None


def _module_for_src_file(path: Path, src_root: Path) -> str | None:
    rel = path.relative_to(src_root)
    return rel.parts[0] if len(rel.parts) >= 2 else None


def check_repository(
    *,
    arch_doc: Path = ARCH_DOC,
    public_root: Path = PUBLIC_ROOT,
    src_root: Path = SRC_ROOT,
    tests_root: Path = TESTS_ROOT,
) -> list[str]:
    """Validate include graph and third-party isolation across public, src, and tests."""
    allowed_deps, errors = parse_allowed_deps(arch_doc)
    if errors:
        return errors

    if public_root.exists():
        for header in sorted(public_root.rglob("*.hpp")):
            rel_posix = header.as_posix()
            mod = _module_for_public_header(header, public_root)
            if mod is None or mod not in allowed_deps:
                errors.append(f"{rel_posix}: public header sits outside a known module directory")
                continue

            header_lines = header.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(header_lines, start=1):
                inc_match = INCLUDE_RE.match(line)
                if not inc_match:
                    continue
                target = inc_match.group(2)
                if target.startswith(("Eigen/", "eigen3/", "nlohmann/")) or target == "omp.h":
                    errors.append(
                        f"{rel_posix}:{line_no}: forbidden third-party include "
                        f"'<{target}>' in public header"
                    )
                if "detail/" in target or target.startswith("src/"):
                    errors.append(
                        f"{rel_posix}:{line_no}: forbidden internal include "
                        f"'{target}' in public header"
                    )
                if mod_match := CXXGPR_INCLUDE_RE.match(target):
                    target_mod = mod_match.group(1)
                    if target_mod != mod and target_mod not in allowed_deps[mod]:
                        errors.append(
                            f"{rel_posix}:{line_no}: disallowed dependency "
                            f"'{mod}' -> '{target_mod}'"
                        )

    if src_root.exists():
        for src_file in sorted(
            p for p in src_root.rglob("*") if p.is_file() and p.suffix in {".hpp", ".cpp"}
        ):
            rel_posix = src_file.as_posix()
            rel_under_src = src_file.relative_to(src_root).as_posix()
            mod = _module_for_src_file(src_file, src_root)
            if mod is None or mod not in allowed_deps:
                errors.append(f"{rel_posix}: source file sits outside a known module directory")
                continue

            src_lines = src_file.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(src_lines, start=1):
                inc_match = INCLUDE_RE.match(line)
                if not inc_match:
                    continue
                target = inc_match.group(2)
                if mod_match := CXXGPR_INCLUDE_RE.match(target):
                    target_mod = mod_match.group(1)
                    if target_mod != mod and target_mod not in allowed_deps[mod]:
                        errors.append(
                            f"{rel_posix}:{line_no}: disallowed dependency "
                            f"'{mod}' -> '{target_mod}'"
                        )
                if (
                    target.startswith(("Eigen/", "eigen3/"))
                    and mod not in EIGEN_MODULES
                    and rel_under_src != "core/build_info.cpp"
                ):
                    errors.append(
                        f"{rel_posix}:{line_no}: Eigen include not permitted in module '{mod}'"
                    )
                if target == "omp.h" and mod not in OPENMP_MODULES:
                    errors.append(
                        f"{rel_posix}:{line_no}: OpenMP include not permitted in module '{mod}'"
                    )
                if (
                    target.startswith("nlohmann/")
                    and mod not in JSON_MODULES
                    and rel_under_src != "core/build_info.cpp"
                ):
                    errors.append(
                        f"{rel_posix}:{line_no}: nlohmann/json include "
                        f"not permitted in module '{mod}'"
                    )

    if tests_root.exists():
        for test_file in sorted(
            p for p in tests_root.rglob("*") if p.is_file() and p.suffix in {".hpp", ".cpp"}
        ):
            rel_posix = test_file.as_posix()
            for line_no, line in enumerate(
                test_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                inc_match = INCLUDE_RE.match(line)
                if not inc_match:
                    continue
                target = inc_match.group(2)
                if target.startswith("src/"):
                    errors.append(
                        f"{rel_posix}:{line_no}: tests must not include "
                        f"private src/ path '{target}'"
                    )

    return errors


def _self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pub = root / "include" / "cxxgpr"
        src = root / "src"
        (pub / "core").mkdir(parents=True)
        (src / "kernel").mkdir(parents=True)
        (pub / "core" / "bad.hpp").write_text("#include <Eigen/Core>\n", encoding="utf-8")
        (src / "kernel" / "bad.cpp").write_text('#include "cxxgpr/gpr/fit.hpp"\n', encoding="utf-8")
        errs = check_repository(
            arch_doc=ARCH_DOC,
            public_root=pub,
            src_root=src,
            tests_root=root / "tests",
        )
        if len(errs) != 2:
            print(f"Self-test expected 2 errors, got {len(errs)}: {errs}")
            return 1
    print("Architecture check self-test passed.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="Run built-in violation check test."
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    errors = check_repository()
    for error in errors:
        print(error)
    if not errors:
        print("Architecture boundary check passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
