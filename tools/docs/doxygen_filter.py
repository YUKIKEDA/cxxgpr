# SPDX-License-Identifier: Apache-2.0
"""Doxygen input filter: blank out ``extern template`` declarations.

Explicit-instantiation declarations are implementation detail of the header-only templates; Breathe
cannot render them. Line numbers are preserved by keeping the newlines.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXTERN_TEMPLATE = re.compile(r"^[ \t]*extern\s+template\b[^;]*;", re.MULTILINE | re.DOTALL)


def _blank(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", "", match.group(0))


def main() -> None:
    """Print the file given as ``argv[1]`` without ``extern template`` declarations."""
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    sys.stdout.write(_EXTERN_TEMPLATE.sub(_blank, text))


if __name__ == "__main__":
    main()
