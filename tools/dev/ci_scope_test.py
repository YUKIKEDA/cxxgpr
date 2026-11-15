# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/dev/ci_scope.py."""

from __future__ import annotations

import unittest

from ci_scope import select, test_scope, verify_needs


class CiScopeTest(unittest.TestCase):
    def test_docs_only_skips_build_and_benchmark(self) -> None:
        flags = select(["README.md"])
        self.assertFalse(flags["build"])
        self.assertFalse(flags["benchmark"])
        self.assertTrue(flags["doc"])
        self.assertTrue(flags["checks"])

    def test_cpp_change_triggers_build_bench_format_and_arch(self) -> None:
        flags = select(["src/core/build_info.cpp"])
        self.assertTrue(flags["build"])
        self.assertTrue(flags["benchmark"])
        self.assertTrue(flags["format"])
        self.assertTrue(flags["arch"])
        self.assertTrue(flags["checks"])

    def test_skills_change_triggers_skills_only(self) -> None:
        flags = select([".agents/skills/grilling/SKILL.md"])
        self.assertFalse(flags["build"])
        self.assertTrue(flags["skills"])
        self.assertTrue(flags["checks"])

    def test_benchmark_change_triggers_benchmark_and_arch(self) -> None:
        flags = select(["benchmarks/exact_bench.cpp"])
        self.assertTrue(flags["benchmark"])
        self.assertTrue(flags["arch"])
        self.assertTrue(flags["format"])
        self.assertTrue(flags["checks"])
        self.assertFalse(flags["build"])

    def test_full_forces_everything(self) -> None:
        flags = select([], full=True)
        self.assertTrue(all(flags.values()))

    def test_test_scope_single_module_narrows_regex(self) -> None:
        scope = test_scope(["tests/core/build_info_test.cpp"])
        self.assertEqual(scope["test_regex"], "^core\\.")
        self.assertEqual(scope["cpp_files"], "tests/core/build_info_test.cpp")

    def test_test_scope_header_change_runs_all_tests(self) -> None:
        scope = test_scope(["include/cxxgpr/core/build_info.hpp"])
        self.assertEqual(scope["test_regex"], "")
        self.assertEqual(scope["format_files"], "include/cxxgpr/core/build_info.hpp")

    def test_verify_needs_accepts_matching_results(self) -> None:
        payload: dict[str, object] = {
            "changes": {
                "result": "success",
                "outputs": {"build": "true", "benchmark": "false", "checks": "true"},
            },
            "windows": {"result": "success"},
            "linux": {"result": "success"},
            "benchmark": {"result": "skipped"},
            "checks": {"result": "success"},
        }
        self.assertEqual(verify_needs(payload), [])

    def test_verify_needs_fails_closed_on_missing_or_failed_job(self) -> None:
        payload: dict[str, object] = {
            "changes": {
                "result": "success",
                "outputs": {"build": "true", "benchmark": "false", "checks": "true"},
            },
            "windows": {"result": "failure"},
            "linux": {"result": "success"},
            "benchmark": {"result": "skipped"},
            "checks": {"result": "success"},
        }
        errors = verify_needs(payload)
        self.assertTrue(any("windows:" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
