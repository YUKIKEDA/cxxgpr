# SPDX-License-Identifier: Apache-2.0
"""Verify that the user documentation covers every public API symbol, in every language.

Two checks run against the Doxygen XML of ``include/cxxgpr/``:

1. Reference completeness: every public header has a ``doxygenfile`` directive in
   ``docs/<lang>/reference/api/``.
2. Usage coverage: every public symbol (class, struct, concept, alias, enumeration, enumerator,
   free function, variable and member function) appears by its simple name inside inline code or
   a code block of the hand-written tutorials, how-to guides or explanation pages.

Run ``doxygen docs/Doxyfile`` first (``just docs-xml``).
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

XML_DIR = Path("out/docs/doxygen/xml")
INCLUDE_DIR = Path("include/cxxgpr")
DOCS_DIR = Path("docs")
LANGS = ("en", "ja")
GUIDE_SECTIONS = ("tutorials", "how-to", "explanation")

CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"^(`{3,}|~{3,}|:{3,})[^\n]*\n(.*?)^\1\s*$", re.DOTALL | re.MULTILINE)
WORD_RE = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class Symbol:
    """One documented public symbol."""

    name: str
    kind: str
    scope: str


def _text(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext()).strip()


def public_symbols(xml_dir: Path) -> list[Symbol]:
    """Collect public, non-``detail`` symbols from Doxygen XML."""
    found: dict[tuple[str, str], Symbol] = {}

    def add(name: str, kind: str, scope: str) -> None:
        name = name.split("<", 1)[0].strip()  # explicit specialisations: ``f< T >`` -> ``f``
        if not name or name.startswith(("operator", "~", "@")):
            return
        found.setdefault((name, kind), Symbol(name, kind, scope))

    for path in sorted(xml_dir.glob("*.xml")):
        if path.name.startswith(("index", "Doxyfile", "combine")):
            continue
        root = ET.parse(path).getroot()
        for comp in root.findall("compounddef"):
            kind = comp.get("kind", "")
            cname = _text(comp.find("compoundname"))
            if kind not in {"namespace", "class", "struct", "union", "concept"}:
                continue
            if "detail" in cname.split("::") or cname == "std" or "internal" in cname.lower():
                continue
            if comp.get("prot") in {"private", "protected"}:
                continue
            simple = cname.rsplit("::", 1)[-1]
            if kind != "namespace":
                add(simple, kind, cname)
            is_class = kind in {"class", "struct", "union"}
            for section in comp.findall("sectiondef"):
                for member in section.findall("memberdef"):
                    if member.get("prot") not in {None, "public"}:
                        continue
                    mkind = member.get("kind", "")
                    mname = _text(member.find("name"))
                    if mkind == "friend":
                        continue
                    if is_class and mkind == "function" and mname == simple:
                        continue
                    add(mname, mkind, cname)
                    if mkind == "enum":
                        for value in member.findall("enumvalue"):
                            add(_text(value.find("name")), "enumvalue", f"{cname}::{mname}")
    return sorted(found.values(), key=lambda s: (s.kind, s.name))


def guide_words(lang: str) -> set[str]:
    """Identifiers that appear in inline code or code blocks of the hand-written guides."""
    words: set[str] = set()
    for section in GUIDE_SECTIONS:
        for page in (DOCS_DIR / lang / section).rglob("*.md"):
            text = page.read_text(encoding="utf-8")
            for span in CODE_SPAN_RE.findall(text):
                words.update(WORD_RE.findall(span))
            for _fence, body in FENCE_RE.findall(text):
                words.update(WORD_RE.findall(body))
    return words


def missing_reference(lang: str) -> list[str]:
    """Headers without a doxygenfile directive in the generated API reference."""
    api_dir = DOCS_DIR / lang / "reference" / "api"
    text = (
        "\n".join(p.read_text(encoding="utf-8") for p in api_dir.glob("*.md"))
        if api_dir.is_dir()
        else ""
    )
    named = set(re.findall(r"\{doxygenfile\}\s+(\S+)", text))
    umbrella = {"kernel.hpp", "optimizer.hpp", "transform.hpp", "workspace.hpp"}
    headers = sorted(p.name for p in INCLUDE_DIR.rglob("*.hpp") if p.name not in umbrella)
    return [h for h in headers if h not in named]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=XML_DIR)
    parser.add_argument(
        "--show", type=int, default=60, help="Maximum missing names printed per language."
    )
    args = parser.parse_args()

    if not args.xml.is_dir():
        print(f"ERROR: Doxygen XML not found at {args.xml}; run `just docs-xml` first.")
        return 2
    symbols = public_symbols(args.xml)
    failed = False
    for lang in LANGS:
        words = guide_words(lang)
        missing = [s for s in symbols if s.name not in words]
        no_ref = missing_reference(lang)
        covered = len(symbols) - len(missing)
        print(
            f"[{lang}] {covered}/{len(symbols)} public symbols covered by guides; "
            f"{len(no_ref)} headers missing in reference"
        )
        for header in no_ref:
            print(f"  reference missing: {header}")
        for sym in missing[: args.show]:
            where = (
                sym.scope
                if sym.kind in {"class", "struct", "union", "concept"}
                else f"{sym.scope}::{sym.name}"
            )
            print(f"  not documented: {sym.kind} {where}")
        if len(missing) > args.show:
            print(f"  ... and {len(missing) - args.show} more")
        failed = failed or bool(missing) or bool(no_ref)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
