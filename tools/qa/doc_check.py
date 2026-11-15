# SPDX-License-Identifier: Apache-2.0
"""Verify English public Doxygen coverage, zero internal ID leaks, and Markdown links."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

PUBLIC_INCLUDE_ROOT = Path("include/cxxgpr")

INTERNAL_ID_RE = re.compile(r"\b(?:REQ|DES|FMT|QA)-\d+\b|\bM[0-6]-\d+\b")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
DECL_RE = re.compile(
    r"^\s*(?:template\s*<.*?>\s*)?(?:inline\s+|constexpr\s+|static\s+|friend\s+|virtual\s+|explicit\s+)*"
    r"(?:struct|class|enum(?:\s+class)?|using\s+[A-Za-z_]\w*\s*=)\b"
)
FUNC_DECL_RE = re.compile(
    r"^\s*(?:\[\[[^\]]+\]\]\s*)?(?:template\s*<.*?>\s*)?(?:inline\s+|constexpr\s+|explicit\s+)*"
    r"[A-Za-z_][\w:<>,\s\*&]*\s+[A-Za-z_]\w*\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?;"
)
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LOCAL_ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:(?:\\\\|\\)(?:Users|home)(?:\\\\|\\)", re.IGNORECASE
)


def check_public_headers(root: Path = PUBLIC_INCLUDE_ROOT) -> list[str]:
    """Verify public headers have English Doxygen comments and zero internal spec IDs."""
    if not root.exists():
        return []
    errors: list[str] = []
    for header in sorted(root.rglob("*.hpp")):
        rel = header.as_posix()
        lines = header.read_text(encoding="utf-8").splitlines()
        in_block_comment = False
        saw_doxygen = False
        brace_depth = 0
        pending_namespace_brace = False

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            if INTERNAL_ID_RE.search(line):
                errors.append(
                    f"{rel}:{line_no}: internal spec/milestone ID leaked into public header"
                )
            if CJK_RE.search(line):
                errors.append(f"{rel}:{line_no}: non-English text found in public header")

            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith("/**"):
                saw_doxygen = True
                if "*/" not in stripped:
                    in_block_comment = True
                continue
            if stripped.startswith("/*"):
                if "*/" not in stripped:
                    in_block_comment = True
                continue
            if stripped.startswith("///"):
                saw_doxygen = True
                continue
            if stripped.startswith("//") or stripped.startswith("#") or not stripped:
                continue

            if stripped.startswith("[[") and not stripped.endswith(";"):
                # Attribute line above a declaration retains pending Doxygen state.
                if "{" in stripped:
                    brace_depth += stripped.count("{") - stripped.count("}")
                continue

            # Track namespace braces at depth 0/1 so namespace members count as top-level API.
            if stripped.startswith("namespace "):
                if "{" in stripped:
                    brace_depth += stripped.count("{") - stripped.count("}")
                else:
                    pending_namespace_brace = True
                saw_doxygen = False
                continue

            if pending_namespace_brace:
                if "{" in stripped:
                    brace_depth += stripped.count("{") - stripped.count("}")
                    pending_namespace_brace = False
                continue

            at_api_level = brace_depth <= 1
            if at_api_level and (DECL_RE.match(stripped) or FUNC_DECL_RE.match(stripped)):
                if not saw_doxygen:
                    errors.append(
                        f"{rel}:{line_no}: public declaration missing English "
                        "Doxygen comment ('///' or '/**')"
                    )
                saw_doxygen = False

            if "{" in stripped or "}" in stripped:
                brace_depth += stripped.count("{") - stripped.count("}")
                if "}" in stripped:
                    saw_doxygen = False
            elif stripped.endswith(";"):
                saw_doxygen = False

    return errors


def _iter_doc_files() -> list[Path]:
    docs: list[Path] = []
    for path in sorted(Path().rglob("*")):
        if not path.is_file():
            continue
        parts = path.parts
        if any(p.startswith(".") and p not in {".github", ".cursor"} for p in parts):
            continue
        if any(p in {"out", "build"} for p in parts):
            continue
        if path.suffix in {".md", ".mdc"}:
            docs.append(path)
    return docs


def check_markdown_links() -> list[str]:
    """Verify all relative Markdown links resolve to existing files in the repository."""
    errors: list[str] = []
    for doc in _iter_doc_files():
        rel = doc.as_posix()
        in_fence = False
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if LOCAL_ABS_PATH_RE.search(line):
                errors.append(f"{rel}:{line_no}: local machine path found in documentation")
            for raw_target in MD_LINK_RE.findall(line):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = unquote(target.split("#", 1)[0])
                if not path_part:
                    continue
                resolved = (doc.parent / path_part).resolve()
                if not resolved.exists():
                    errors.append(f"{rel}:{line_no}: broken relative link '{raw_target}'")
    return errors


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="Run built-in header check self-test."
    )
    args = parser.parse_args()

    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            pub = Path(tmp) / "include" / "cxxgpr"
            pub.mkdir(parents=True)
            bad = pub / "bad.hpp"
            bad.write_text("/// REQ-01 leak.\nstruct Bad {};\nstruct Undoc {};\n", encoding="utf-8")
            errs = check_public_headers(pub)
            if len(errs) != 2:
                print(f"Self-test expected 2 errors, got {len(errs)}: {errs}")
                return 1
        print("Doc check self-test passed.")
        return 0

    errors = [*check_public_headers(), *check_markdown_links()]
    for error in errors:
        print(error)
    if not errors:
        print("Documentation check passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
