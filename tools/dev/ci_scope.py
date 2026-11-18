# SPDX-License-Identifier: Apache-2.0
"""Select affected CI jobs and tests from changed repository paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import cast

BUILD_ROOT_FILES = frozenset(
    {
        "CMakeLists.txt",
        "CMakePresets.json",
        "vcpkg.json",
        "vcpkg-configuration.json",
        ".clang-tidy",
        ".clang-format",
    }
)
PYTHON_CONFIG_FILES = frozenset({"ruff.toml", "pyrightconfig.json", "pyproject.toml"})
BUILD_PREFIXES = ("include/", "src/", "tests/", "qa/", "cmake/")
BENCH_PREFIXES = ("benchmarks/", "include/", "src/", "cmake/")
CPP_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"})


def object_map(value: object) -> dict[str, object]:
    """Narrow a JSON object to a typed dict."""
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def changed_paths(base: str, head: str) -> list[str]:
    """Query git diff for changed POSIX paths between base and head commits."""
    if not base or not head:
        return []
    proc = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _is_cpp_source(path: str) -> bool:
    return Path(path).suffix.lower() in CPP_SUFFIXES


def _is_markdown(path: str) -> bool:
    return path.endswith(".md") or path.endswith(".mdc")


def select(paths: list[str], *, full: bool = False) -> dict[str, bool]:
    """Compute CI job enable flags from changed repository paths."""
    flags = {
        "build": full,
        "benchmark": full,
        "format": full,
        "python": full,
        "skills": full,
        "trace": full,
        "arch": full,
        "doc": full,
    }
    if full:
        flags["checks"] = True
        return flags

    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        if (
            path in BUILD_ROOT_FILES
            or path.startswith(BUILD_PREFIXES)
            or path in {"tools/dev/msvc.ps1", "tools/qa/results.py", "THIRD_PARTY_NOTICES.md"}
            or path.startswith(".github/workflows/")
        ):
            flags["build"] = True

        if (
            path in BUILD_ROOT_FILES
            or path.startswith(BENCH_PREFIXES)
            or path.startswith(".github/workflows/")
        ):
            flags["benchmark"] = True

        if _is_cpp_source(path) or path in {".clang-format", ".clang-tidy"}:
            flags["format"] = True

        if path.endswith(".py") or path in PYTHON_CONFIG_FILES:
            flags["python"] = True

        if path.startswith((".agents/skills/", ".claude/skills/")) or path in {
            "tools/dev/sync_skills.py",
            "skills-lock.json",
        }:
            flags["skills"] = True

        if path in {
            "docs/design.md",
            "docs/persist-format.md",
            "docs/quality/quality-plan.md",
            "docs/quality/traceability.md",
            "tools/qa/trace.py",
        } or path.startswith(("docs/verification/", "tests/")):
            flags["trace"] = True

        if path in {"docs/architecture.md", "tools/qa/arch_check.py"} or path.startswith(
            ("include/", "src/", "tests/", "benchmarks/")
        ):
            flags["arch"] = True

        if (
            path in {"docs/conventions.md", "tools/qa/doc_check.py"}
            or path.startswith(("include/", ".cursor/rules/"))
            or _is_markdown(path)
        ):
            flags["doc"] = True

        if path.startswith(".github/workflows/"):
            flags["format"] = True
            flags["python"] = True
            flags["skills"] = True
            flags["trace"] = True
            flags["arch"] = True
            flags["doc"] = True

    flags["checks"] = any(
        flags[key] for key in ("format", "python", "skills", "trace", "arch", "doc")
    )
    return flags


def test_scope(paths: list[str], *, full: bool = False) -> dict[str, str]:
    """Compute filtered CTest regex, build targets, and C++ file lists."""
    if full:
        return {"test_regex": "", "build_targets": "", "cpp_files": "", "format_files": ""}

    cpp_roots = ("include/", "src/", "tests/", "qa/", "benchmarks/")
    tu_roots = ("src/", "tests/", "qa/")
    normalized_paths = [raw_path.replace("\\", "/") for raw_path in paths]
    cpp_files = sorted(
        {path for path in normalized_paths if _is_cpp_source(path) and path.startswith(cpp_roots)}
    )
    tu_files = [path for path in cpp_files if path.endswith(".cpp") and path.startswith(tu_roots)]
    force_all_format = ".clang-format" in normalized_paths
    force_all_cpp = any(
        path in BUILD_ROOT_FILES or path.startswith("cmake/") for path in normalized_paths
    )
    format_str = "" if force_all_format else " ".join(cpp_files)
    cpp_str = (
        " ".join(tu_files)
        if not force_all_cpp and len(tu_files) == len(cpp_files) and cpp_files
        else ""
    )

    # Until per-module suites grow, narrow CTest only for single-module tests/ edits.
    modules: set[str] = set()
    run_all = False
    for path in normalized_paths:
        if path.startswith("tests/"):
            parts = Path(path).parts
            if len(parts) >= 3 and parts[1] != "support":
                modules.add(parts[1])
            else:
                run_all = True
        elif (
            path in BUILD_ROOT_FILES
            or path.startswith(("include/", "src/", "qa/", "cmake/", ".github/workflows/"))
            or path == "THIRD_PARTY_NOTICES.md"
        ):
            run_all = True

    if run_all or not modules or len(modules) > 1:
        test_regex = ""
    else:
        only_mod = next(iter(modules))
        test_regex = f"^{only_mod}\\."

    return {
        "test_regex": test_regex,
        "build_targets": "",
        "cpp_files": cpp_str,
        "format_files": format_str,
    }


def verify_needs(value: object) -> list[str]:
    """Fail closed if any required downstream CI job failed or mismatched selection."""
    needs = object_map(value)
    if not needs:
        return ["Missing CI_NEEDS_JSON payload"]
    changes = object_map(needs.get("changes"))
    if changes.get("result") != "success":
        return [f"changes job did not succeed: {changes.get('result')}"]
    outputs = object_map(changes.get("outputs"))
    required_flags = ("build", "benchmark", "checks")
    if any(outputs.get(flag) not in ("true", "false") for flag in required_flags):
        return ["Missing or invalid selection outputs"]

    errors: list[str] = []
    for job, flag in (
        ("windows", "build"),
        ("linux", "build"),
        ("benchmark", "benchmark"),
        ("checks", "checks"),
    ):
        expected = "success" if outputs[flag] == "true" else "skipped"
        actual = object_map(needs.get(job)).get("result")
        if actual != expected:
            errors.append(f"{job}: expected {expected}, got {actual}")
    return errors


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-needs", action="store_true")
    args = parser.parse_args()

    if args.verify_needs:
        value: object = json.loads(os.environ["CI_NEEDS_JSON"])
        errors = verify_needs(value)
        print(json.dumps({"errors": errors}))
        return int(bool(errors))

    base = os.environ.get("CI_BASE_SHA", "")
    head = os.environ.get("CI_HEAD_SHA", "")
    full = os.environ.get("CI_FORCE_FULL") == "true" or not base or set(base) == {"0"}
    paths = [] if full else changed_paths(base, head)
    selected = select(paths, full=full)
    scope = test_scope(paths, full=full)
    print(json.dumps({"paths": paths, "selected": selected, "test_scope": scope}, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.writelines(f"{key}={str(value).lower()}\n" for key, value in selected.items())
            stream.writelines(f"{key}={value}\n" for key, value in scope.items())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
