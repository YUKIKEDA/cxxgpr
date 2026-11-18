# SPDX-License-Identifier: Apache-2.0
"""Verify English Doxygen coverage, @-command completeness, inline comments, and Markdown links."""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

CPP_ROOTS: tuple[tuple[Path, str], ...] = (
    (Path("include/cxxgpr"), "header"),
    (Path("src"), "src"),
    (Path("tests"), "test"),
    (Path("benchmarks"), "test"),
    (Path("qa"), "test"),
)

INTERNAL_ID_RE = re.compile(r"\b(?:REQ|DES|FMT|QA)-\d+\b|\bM[0-6]-(?:\d+|[A-Z])\b|\bREL-\d+\b")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LOCAL_ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:(?:\\\\|\\)(?:Users|home)(?:\\\\|\\)", re.IGNORECASE
)

DECL_SPECIFIER_SKIP = frozenset(
    {
        "inline",
        "constexpr",
        "consteval",
        "constinit",
        "static",
        "virtual",
        "explicit",
        "friend",
        "extern",
        "mutable",
        "thread_local",
        "const",
        "volatile",
        "typename",
    }
)

TYPE_MODIFIER_WORDS = frozenset(
    {
        "const",
        "volatile",
        "struct",
        "class",
        "enum",
        "union",
        "typename",
        "signed",
        "unsigned",
        "short",
        "long",
    }
)

BUILTIN_TYPE_WORDS = frozenset(
    {
        "void",
        "bool",
        "char",
        "wchar_t",
        "char8_t",
        "char16_t",
        "char32_t",
        "short",
        "int",
        "long",
        "float",
        "double",
        "auto",
    }
)

MACRO_SKIP_CALLS = frozenset(
    {
        "BENCHMARK",
        "BENCHMARK_TEMPLATE",
        "INSTANTIATE_TEST_SUITE_P",
        "INSTANTIATE_TYPED_TEST_SUITE_P",
        "FRIEND_TEST",
    }
)

TEST_MACROS = frozenset({"TEST", "TEST_F", "TEST_P"})

MULTI_OPS = (
    "<=>",
    "::",
    "->",
    "<<",
    ">>",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "++",
    "--",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "...",
)


@dataclass(frozen=True)
class Token:
    """Lexical token emitted from C++ source text."""

    kind: str  # PREPROC, DOX, COMMENT, BLANK, IDENT, OP, LIT
    text: str
    line: int


@dataclass
class Scope:
    """Enclosing lexical scope during C++ declaration scanning."""

    kind: str  # global, namespace, class, enum, function_body, braced_init
    name: str = ""
    brace_depth: int = 0
    seen_enum_doc: bool = False
    enum_expect_value: bool = False
    enum_paren_depth: int = 0
    enum_bracket_depth: int = 0
    enum_brace_depth: int = 0


def _file_mode(path: Path, default_mode: str) -> str:
    if path.suffix == ".hpp":
        return "header"
    return default_mode


def _is_raw_string_start(line: str, idx: int) -> str | None:
    for prefix in ('R"', 'u8R"', 'uR"', 'UR"', 'LR"'):
        if line.startswith(prefix, idx):
            return prefix
    return None


def _tokenize_cpp(text: str, rel: str, errors: list[str]) -> list[Token]:
    """Tokenize C++ source into structural tokens, Doxygen blocks, and comments."""
    lines = text.splitlines()
    tokens: list[Token] = []
    in_block_comment = False
    in_raw_string = False
    raw_delim = ""
    dox_lines: list[str] = []
    dox_start_line = 0

    def flush_dox() -> None:
        nonlocal dox_lines, dox_start_line
        if dox_lines:
            tokens.append(Token("DOX", "\n".join(dox_lines), dox_start_line))
            dox_lines = []
            dox_start_line = 0

    for line_no, line in enumerate(lines, start=1):
        n = len(line)
        i = 0

        if in_raw_string:
            end_marker = f'){raw_delim}"'
            pos = line.find(end_marker)
            if pos == -1:
                continue
            in_raw_string = False
            i = pos + len(end_marker)
            tokens.append(Token("LIT", '""', line_no))

        if in_block_comment:
            end_pos = line.find("*/")
            if end_pos == -1:
                continue
            in_block_comment = False
            i = end_pos + 2

        while i < n and line[i] in " \t\r":
            i += 1

        if i >= n:
            if not in_block_comment and not in_raw_string:
                flush_dox()
                tokens.append(Token("BLANK", "", line_no))
            continue

        if i < n and line[i] == "#" and tokens_on_line_empty(tokens, line_no):
            flush_dox()
            tokens.append(Token("PREPROC", line[i:].strip(), line_no))
            continue

        while i < n:
            ch = line[i]
            if ch in " \t\r":
                i += 1
                continue

            if line.startswith("///", i):
                if not dox_lines:
                    dox_start_line = line_no
                dox_lines.append(line[i + 3 :].strip())
                break

            if line.startswith("//", i):
                flush_dox()
                tokens.append(Token("COMMENT", line[i:].strip(), line_no))
                break

            if line.startswith("/*", i):
                flush_dox()
                if line.startswith(("/**", "/*!"), i) and not line.startswith("/**/", i):
                    errors.append(
                        f"{rel}:{line_no}: use '///' Doxygen blocks instead of '/**' or '/*!'"
                    )
                end_pos = line.find("*/", i + 2)
                if end_pos == -1:
                    in_block_comment = True
                    break
                i = end_pos + 2
                continue

            flush_dox()

            raw_prefix = _is_raw_string_start(line, i)
            if raw_prefix is not None:
                paren_pos = line.find("(", i + len(raw_prefix))
                if paren_pos != -1:
                    delim = line[i + len(raw_prefix) : paren_pos]
                    end_marker = f'){delim}"'
                    end_pos = line.find(end_marker, paren_pos + 1)
                    tokens.append(Token("LIT", '""', line_no))
                    if end_pos == -1:
                        in_raw_string = True
                        raw_delim = delim
                        break
                    i = end_pos + len(end_marker)
                    continue

            if ch == '"':
                j = i + 1
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == '"':
                        j += 1
                        break
                    j += 1
                tokens.append(Token("LIT", line[i:j], line_no))
                i = j
                continue

            if ch == "'" and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] == "_")):
                j = i + 1
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == "'":
                        j += 1
                        break
                    j += 1
                tokens.append(Token("LIT", line[i:j], line_no))
                i = j
                continue

            if ch.isalpha() or ch == "_":
                j = i + 1
                while j < n and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                tokens.append(Token("IDENT", line[i:j], line_no))
                i = j
                continue

            if ch.isdigit():
                j = i + 1
                while j < n and (line[j].isalnum() or line[j] in "._'"):
                    if line[j] in "eEpP" and j + 1 < n and line[j + 1] in "+-":
                        j += 2
                        continue
                    j += 1
                tokens.append(Token("LIT", line[i:j], line_no))
                i = j
                continue

            matched_op: str | None = None
            for op in MULTI_OPS:
                if line.startswith(op, i):
                    matched_op = op
                    break
            if matched_op is not None:
                tokens.append(Token("OP", matched_op, line_no))
                i += len(matched_op)
                continue

            tokens.append(Token("OP", ch, line_no))
            i += 1

    flush_dox()
    return tokens


def tokens_on_line_empty(tokens: list[Token], line_no: int) -> bool:
    return not tokens or tokens[-1].line != line_no


def _split_top_level(tokens: list[Token], sep: str = ",") -> list[list[Token]]:
    """Split tokens by separator at bracket/paren/angle/brace depth 0."""
    chunks: list[list[Token]] = []
    cur: list[Token] = []
    paren = 0
    bracket = 0
    brace = 0
    angle = 0
    for tok in tokens:
        if tok.kind == "OP":
            if tok.text == "(":
                paren += 1
            elif tok.text == ")":
                paren = max(0, paren - 1)
            elif tok.text == "[":
                bracket += 1
            elif tok.text == "]":
                bracket = max(0, bracket - 1)
            elif tok.text == "{":
                brace += 1
            elif tok.text == "}":
                brace = max(0, brace - 1)
            elif tok.text == "<" and paren == 0 and bracket == 0 and brace == 0:
                angle += 1
            elif tok.text == ">" and paren == 0 and bracket == 0 and brace == 0:
                angle = max(0, angle - 1)
            elif tok.text == ">>" and paren == 0 and bracket == 0 and brace == 0 and angle >= 2:
                angle -= 2
            elif tok.text == sep and paren == 0 and bracket == 0 and brace == 0 and angle == 0:
                chunks.append(cur)
                cur = []
                continue
        cur.append(tok)
    if cur:
        chunks.append(cur)
    return chunks


def _extract_tparam_names(tmpl_tokens: list[Token]) -> list[str]:
    """Extract parameter names from a `template <...>` token slice (inside outer < >)."""
    names: list[str] = []
    for chunk in _split_top_level(tmpl_tokens, ","):
        # Strip default template argument `= ...`.
        head: list[Token] = []
        paren = 0
        bracket = 0
        brace = 0
        angle = 0
        for tok in chunk:
            if tok.kind == "OP":
                if tok.text == "(":
                    paren += 1
                elif tok.text == ")":
                    paren = max(0, paren - 1)
                elif tok.text == "[":
                    bracket += 1
                elif tok.text == "]":
                    bracket = max(0, bracket - 1)
                elif tok.text == "{":
                    brace += 1
                elif tok.text == "}":
                    brace = max(0, brace - 1)
                elif tok.text == "<":
                    angle += 1
                elif tok.text == ">":
                    angle = max(0, angle - 1)
                elif tok.text == "=" and paren == 0 and bracket == 0 and brace == 0 and angle == 0:
                    break
            head.append(tok)
        if not head:
            continue
        last = head[-1]
        if last.kind == "IDENT" and last.text not in {"typename", "class"}:
            names.append(last.text)
    return names


def _extract_param_names(param_tokens: list[Token]) -> list[str]:
    """Extract named parameter identifiers from a function parameter list `( ... )`."""
    if not param_tokens:
        return []
    if len(param_tokens) == 1 and param_tokens[0].text == "void":
        return []
    names: list[str] = []
    for chunk in _split_top_level(param_tokens, ","):
        if not chunk:
            continue
        if len(chunk) == 1 and chunk[0].kind == "OP" and chunk[0].text == "...":
            continue
        # Strip attributes like [[maybe_unused]].
        filtered: list[Token] = []
        i = 0
        while i < len(chunk):
            if (
                chunk[i].kind == "OP"
                and chunk[i].text == "["
                and i + 1 < len(chunk)
                and chunk[i + 1].kind == "OP"
                and chunk[i + 1].text == "["
            ):
                bdepth = 0
                while i < len(chunk):
                    if chunk[i].kind == "OP" and chunk[i].text == "[":
                        bdepth += 1
                    elif chunk[i].kind == "OP" and chunk[i].text == "]":
                        bdepth -= 1
                        if bdepth == 0:
                            i += 1
                            break
                    i += 1
                continue
            filtered.append(chunk[i])
            i += 1

        # Strip `= default_value`.
        head: list[Token] = []
        paren = 0
        bracket = 0
        brace = 0
        angle = 0
        for tok in filtered:
            if tok.kind == "OP":
                if tok.text == "(":
                    paren += 1
                elif tok.text == ")":
                    paren = max(0, paren - 1)
                elif tok.text == "[":
                    bracket += 1
                elif tok.text == "]":
                    bracket = max(0, bracket - 1)
                elif tok.text == "{":
                    brace += 1
                elif tok.text == "}":
                    brace = max(0, brace - 1)
                elif tok.text == "<":
                    angle += 1
                elif tok.text == ">":
                    angle = max(0, angle - 1)
                elif tok.text == "=" and paren == 0 and bracket == 0 and brace == 0 and angle == 0:
                    break
            head.append(tok)
        if not head:
            continue

        # Detect function pointer / function reference declarator `(*name)(...)` or `(&name)(...)`.
        fn_ptr_name: str | None = None
        for idx in range(len(head) - 3):
            if (
                head[idx].kind == "OP"
                and head[idx].text == "("
                and head[idx + 1].kind == "OP"
                and head[idx + 1].text in {"*", "&"}
                and head[idx + 2].kind == "IDENT"
                and head[idx + 3].kind == "OP"
                and head[idx + 3].text == ")"
            ):
                fn_ptr_name = head[idx + 2].text
                break
        if fn_ptr_name is not None:
            names.append(fn_ptr_name)
            continue

        # Strip trailing array brackets `name[N]`.
        while len(head) >= 2 and head[-1].kind == "OP" and head[-1].text == "]":
            b = 0
            while head:
                popped = head.pop()
                if popped.kind == "OP" and popped.text == "]":
                    b += 1
                elif popped.kind == "OP" and popped.text == "[":
                    b -= 1
                    if b == 0:
                        break
        if len(head) < 2:
            continue
        last = head[-1]
        prev = head[-2]
        if (
            last.kind != "IDENT"
            or last.text in BUILTIN_TYPE_WORDS
            or last.text in TYPE_MODIFIER_WORDS
        ):
            continue
        if prev.kind == "OP" and prev.text in {"::", "<", ","}:
            continue
        if (
            prev.kind == "IDENT"
            and prev.text in TYPE_MODIFIER_WORDS
            and prev.text not in {"const", "volatile"}
        ):
            continue
        if prev.kind == "IDENT" and prev.text in {"const", "volatile"}:
            # In `const Foo` (len == 2), Foo is the type unless preceded by `*` / `&` / type.
            if len(head) == 2:
                continue
            prev2 = head[-3]
            if prev2.kind == "OP" and prev2.text in {"::", "<", ","}:
                continue
            if prev2.kind == "IDENT" and prev2.text in TYPE_MODIFIER_WORDS:
                continue
        names.append(last.text)
    return names


def _strip_leading_attributes(tokens: list[Token]) -> list[Token]:
    out: list[Token] = []
    i = 0
    seen_code = False
    while i < len(tokens):
        tok = tokens[i]
        if (
            not seen_code
            and tok.kind == "OP"
            and tok.text == "["
            and i + 1 < len(tokens)
            and tokens[i + 1].kind == "OP"
            and tokens[i + 1].text == "["
        ):
            depth = 0
            while i < len(tokens):
                if tokens[i].kind == "OP" and tokens[i].text == "[":
                    depth += 1
                elif tokens[i].kind == "OP" and tokens[i].text == "]":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            continue
        seen_code = True
        out.append(tok)
        i += 1
    return out


def _consume_angle_block(tokens: list[Token], start: int) -> tuple[list[Token], int]:
    """Given `tokens[start]` is `<`, return (inner_tokens, next_index_after_matching_>)."""
    inner: list[Token] = []
    depth = 1
    paren = 0
    bracket = 0
    i = start + 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "OP":
            if tok.text == "(":
                paren += 1
            elif tok.text == ")":
                paren = max(0, paren - 1)
            elif tok.text == "[":
                bracket += 1
            elif tok.text == "]":
                bracket = max(0, bracket - 1)
            elif paren == 0 and bracket == 0:
                if tok.text == "<":
                    depth += 1
                elif tok.text == ">":
                    depth -= 1
                    if depth == 0:
                        return inner, i + 1
                elif tok.text == ">>":
                    if depth <= 2:
                        if depth == 2:
                            inner.append(Token("OP", ">", tok.line))
                        return inner, i + 1
                    depth -= 2
                    inner.append(tok)
                    i += 1
                    continue
        inner.append(tok)
        i += 1
    return inner, i


def _find_top_level_open_paren(tokens: list[Token]) -> int:
    """Find index of function parameter-list `(`, skipping operator/template parens."""
    angle = 0
    bracket = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "IDENT" and tok.text == "operator":
            i += 1
            if i < len(tokens) and tokens[i].kind == "OP" and tokens[i].text == "(":
                # Skip operator() name parens.
                if i + 1 < len(tokens) and tokens[i + 1].kind == "OP" and tokens[i + 1].text == ")":
                    i += 2
                    continue
            elif i < len(tokens) and tokens[i].kind == "OP" and tokens[i].text == "[":
                if i + 1 < len(tokens) and tokens[i + 1].kind == "OP" and tokens[i + 1].text == "]":
                    i += 2
                    continue
            else:
                # Skip operator symbol tokens or conversion type until `(`.
                while i < len(tokens) and not (tokens[i].kind == "OP" and tokens[i].text == "("):
                    i += 1
                return i if i < len(tokens) else -1
            continue
        if tok.kind == "OP":
            if tok.text == "[":
                bracket += 1
            elif tok.text == "]":
                bracket = max(0, bracket - 1)
            elif bracket == 0:
                if tok.text == "<":
                    angle += 1
                elif tok.text == ">":
                    angle = max(0, angle - 1)
                elif tok.text == ">>" and angle >= 2:
                    angle -= 2
                elif tok.text == "(" and angle == 0:
                    return i
        i += 1
    return -1


def _has_top_level_assign_before(tokens: list[Token], limit_idx: int) -> bool:
    angle = 0
    bracket = 0
    paren = 0
    end = limit_idx if limit_idx >= 0 else len(tokens)
    for tok in tokens[:end]:
        if tok.kind == "OP":
            if tok.text == "(":
                paren += 1
            elif tok.text == ")":
                paren = max(0, paren - 1)
            elif tok.text == "[":
                bracket += 1
            elif tok.text == "]":
                bracket = max(0, bracket - 1)
            elif tok.text == "<" and paren == 0 and bracket == 0:
                angle += 1
            elif tok.text == ">" and paren == 0 and bracket == 0:
                angle = max(0, angle - 1)
            elif tok.text == ">>" and paren == 0 and bracket == 0 and angle >= 2:
                angle -= 2
            elif tok.text == "=" and paren == 0 and bracket == 0 and angle == 0:
                return True
    return False


def _validate_doc_block(
    *,
    doc: Token | None,
    decl_line: int,
    rel: str,
    kind_label: str,
    tparams: list[str],
    params: list[str],
    require_return: bool,
    errors: list[str],
) -> None:
    if doc is None:
        errors.append(f"{rel}:{decl_line}: {kind_label} missing Doxygen comment ('/// @brief ...')")
        return
    if "@brief" not in doc.text:
        errors.append(f"{rel}:{doc.line}: {kind_label} Doxygen block missing '@brief'")
    for tp in tparams:
        if not re.search(rf"@tparam\s+{re.escape(tp)}\b", doc.text):
            errors.append(f"{rel}:{doc.line}: {kind_label} Doxygen block missing '@tparam {tp}'")
    for param in params:
        if not re.search(rf"@param(?:\[(?:in|out|in,out)\])?\s+{re.escape(param)}\b", doc.text):
            errors.append(f"{rel}:{doc.line}: {kind_label} Doxygen block missing '@param {param}'")
    if require_return and not re.search(r"@returns?\b", doc.text):
        errors.append(f"{rel}:{doc.line}: {kind_label} Doxygen block missing '@return'")


def check_cpp_file(path: Path, default_mode: str) -> list[str]:
    """Validate Doxygen coverage, @-commands, inline comments, and language for a C++ file."""
    rel = path.as_posix()
    mode = _file_mode(path, default_mode)
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        if CJK_RE.search(line):
            errors.append(f"{rel}:{line_no}: non-English text found in C++ file")
        if LOCAL_ABS_PATH_RE.search(line):
            errors.append(f"{rel}:{line_no}: local machine path found in C++ file")
        if mode in {"header", "src"} and INTERNAL_ID_RE.search(line):
            errors.append(
                f"{rel}:{line_no}: internal spec/milestone ID leaked into C++ library file"
            )

    tokens: list[Token] = _tokenize_cpp(text, rel, errors)

    # Verify leading @file and @brief Doxygen block before first code declaration.
    saw_file_doc = False
    for tok in tokens:
        if tok.kind in {"PREPROC", "BLANK", "COMMENT"}:
            continue
        if tok.kind == "DOX":
            if re.search(r"(?:^|\s)@file\s+\S+", tok.text) and re.search(
                r"(?:^|\s)@brief\b", tok.text
            ):
                saw_file_doc = True
                break
            continue
        break
    if not saw_file_doc:
        errors.append(
            f"{rel}:1: file missing top-level Doxygen header ('/// @file <name>' and '/// @brief')"
        )

    scopes: list[Scope] = [Scope("global")]
    pending_doc: Token | None = None
    inline_comment_count = 0
    func_body_code_lines: set[int] = set()

    n_tok: int = len(tokens)
    idx: int = 0

    while idx < n_tok:
        tok = tokens[idx]
        cur_scope = scopes[-1]

        if cur_scope.kind == "function_body":
            if tok.kind == "COMMENT" and not tok.text.startswith(
                ("// SPDX", "// NOLINT", "// Spec:")
            ):
                inline_comment_count += 1
            elif tok.kind == "OP" and tok.text == "{":
                cur_scope.brace_depth += 1
            elif tok.kind == "OP" and tok.text == "}":
                cur_scope.brace_depth -= 1
                if cur_scope.brace_depth == 0:
                    scopes.pop()
            elif tok.kind not in {"PREPROC", "BLANK", "DOX"}:
                func_body_code_lines.add(tok.line)
            idx += 1
            continue

        if cur_scope.kind == "braced_init":
            if tok.kind == "OP" and tok.text == "{":
                cur_scope.brace_depth += 1
            elif tok.kind == "OP" and tok.text == "}":
                cur_scope.brace_depth -= 1
                if cur_scope.brace_depth == 0:
                    scopes.pop()
            idx += 1
            continue

        if cur_scope.kind == "enum":
            if tok.kind == "DOX":
                pending_doc = tok
                idx += 1
                continue
            if tok.kind in {"BLANK", "PREPROC"}:
                if (
                    tok.kind == "PREPROC"
                    and not tok.text.startswith(("#if", "#else", "#elif", "#endif"))
                ) or tok.kind == "BLANK":
                    pending_doc = None
                idx += 1
                continue
            if tok.kind == "COMMENT":
                if not tok.text.startswith("// NOLINT"):
                    pending_doc = None
                idx += 1
                continue
            if cur_scope.enum_expect_value:
                if tok.kind == "OP":
                    if tok.text == "(":
                        cur_scope.enum_paren_depth += 1
                    elif tok.text == ")":
                        cur_scope.enum_paren_depth = max(0, cur_scope.enum_paren_depth - 1)
                    elif tok.text == "[":
                        cur_scope.enum_bracket_depth += 1
                    elif tok.text == "]":
                        cur_scope.enum_bracket_depth = max(0, cur_scope.enum_bracket_depth - 1)
                    elif tok.text == "{":
                        cur_scope.enum_brace_depth += 1
                    elif tok.text == "}":
                        if (
                            cur_scope.enum_paren_depth == 0
                            and cur_scope.enum_bracket_depth == 0
                            and cur_scope.enum_brace_depth == 0
                        ):
                            cur_scope.enum_expect_value = False
                            pending_doc = None
                            scopes.pop()
                            # Skip optional trailing `;` after enum closing brace.
                            if (
                                idx + 1 < n_tok
                                and tokens[idx + 1].kind == "OP"
                                and tokens[idx + 1].text == ";"
                            ):
                                idx += 2
                            else:
                                idx += 1
                            continue
                        cur_scope.enum_brace_depth = max(0, cur_scope.enum_brace_depth - 1)
                    elif (
                        tok.text == ","
                        and cur_scope.enum_paren_depth == 0
                        and cur_scope.enum_bracket_depth == 0
                        and cur_scope.enum_brace_depth == 0
                    ):
                        cur_scope.enum_expect_value = False
                        pending_doc = None
                idx += 1
                continue
            if tok.kind == "OP" and tok.text == "}":
                pending_doc = None
                scopes.pop()
                if idx + 1 < n_tok and tokens[idx + 1].kind == "OP" and tokens[idx + 1].text == ";":
                    idx += 2
                else:
                    idx += 1
                continue
            if tok.kind == "IDENT":
                _validate_doc_block(
                    doc=pending_doc,
                    decl_line=tok.line,
                    rel=rel,
                    kind_label=f"enumerator '{tok.text}'",
                    tparams=[],
                    params=[],
                    require_return=False,
                    errors=errors,
                )
                pending_doc = None
                cur_scope.enum_expect_value = True
                idx += 1
                continue
            idx += 1
            continue

        # Scope is global, namespace, or class.
        if tok.kind == "DOX":
            pending_doc = tok
            idx += 1
            continue
        if tok.kind == "COMMENT":
            if not tok.text.startswith(("// NOLINT", "// Spec:")):
                pending_doc = None
            idx += 1
            continue
        if tok.kind in {"BLANK", "PREPROC"}:
            if tok.kind == "BLANK" or tok.text.startswith(("#pragma", "#include", "#define")):
                pending_doc = None
            idx += 1
            continue
        if tok.kind == "OP" and tok.text == "}":
            pending_doc = None
            if len(scopes) > 1:
                scopes.pop()
            if idx + 1 < n_tok and tokens[idx + 1].kind == "OP" and tokens[idx + 1].text == ";":
                idx += 2
            else:
                idx += 1
            continue
        if tok.kind == "OP" and tok.text == ";":
            pending_doc = None
            idx += 1
            continue

        if (
            cur_scope.kind == "class"
            and tok.kind == "IDENT"
            and tok.text in {"public", "protected", "private"}
            and idx + 1 < n_tok
            and tokens[idx + 1].kind == "OP"
            and tokens[idx + 1].text == ":"
        ):
            pending_doc = None
            idx += 2
            continue

        # Parse one complete declaration starting at `idx`.
        decl_doc = pending_doc
        pending_doc = None
        tparams: list[str] = []

        # Consume leading `template <...>`, or skip explicit instantiations.
        if (
            tok.kind == "IDENT"
            and tok.text == "extern"
            and idx + 1 < n_tok
            and tokens[idx + 1].text == "template"
        ):
            while idx < n_tok and not (tokens[idx].kind == "OP" and tokens[idx].text == ";"):
                idx += 1
            idx += 1
            continue

        if tok.kind == "IDENT" and tok.text == "template":
            if idx + 1 < n_tok and tokens[idx + 1].kind == "OP" and tokens[idx + 1].text == "<":
                tmpl_inner, idx = _consume_angle_block(tokens, idx + 1)
                tparams = _extract_tparam_names(tmpl_inner)
                # Skip optional blank/comment/preproc tokens between template<...> and declaration.
                while idx < n_tok and tokens[idx].kind in {"BLANK", "COMMENT", "PREPROC"}:
                    idx += 1
                if idx >= n_tok:
                    break
                tok = tokens[idx]
            else:
                # Explicit instantiation such as `template void foo<int>();`.
                while idx < n_tok and not (tokens[idx].kind == "OP" and tokens[idx].text == ";"):
                    idx += 1
                idx += 1
                continue

        # Collect declaration header tokens until top-level `;` or `{`.
        # First skip static_assert, friend forward declarations, and test/bench macros.
        if tok.kind == "IDENT" and tok.text == "static_assert":
            pdepth = 0
            while idx < n_tok:
                if tokens[idx].kind == "OP" and tokens[idx].text == "(":
                    pdepth += 1
                elif tokens[idx].kind == "OP" and tokens[idx].text == ")":
                    pdepth = max(0, pdepth - 1)
                elif tokens[idx].kind == "OP" and tokens[idx].text == ";" and pdepth == 0:
                    idx += 1
                    break
                idx += 1
            continue

        if tok.kind == "IDENT" and tok.text in MACRO_SKIP_CALLS:
            while idx < n_tok and not (tokens[idx].kind == "OP" and tokens[idx].text == ";"):
                idx += 1
            idx += 1
            continue

        if (
            tok.kind == "IDENT"
            and tok.text == "friend"
            and idx + 1 < n_tok
            and tokens[idx + 1].kind == "IDENT"
            and tokens[idx + 1].text in {"class", "struct", "union"}
        ):
            while idx < n_tok and not (tokens[idx].kind == "OP" and tokens[idx].text == ";"):
                idx += 1
            idx += 1
            continue

        # Check TEST / TEST_F / TEST_P macros in tests.
        if tok.kind == "IDENT" and tok.text in TEST_MACROS:
            _validate_doc_block(
                doc=decl_doc,
                decl_line=tok.line,
                rel=rel,
                kind_label=f"test case '{tok.text}'",
                tparams=[],
                params=[],
                require_return=False,
                errors=errors,
            )
            while idx < n_tok and not (tokens[idx].kind == "OP" and tokens[idx].text == "{"):
                idx += 1
            if idx < n_tok:
                scopes.append(Scope("function_body", brace_depth=1))
                idx += 1
            continue

        # Check namespace declaration.
        if tok.kind == "IDENT" and tok.text == "namespace":
            ns_tokens: list[Token] = []
            j: int = idx + 1
            while j < n_tok and not (tokens[j].kind == "OP" and tokens[j].text in {"{", ";"}):
                ns_tokens.append(tokens[j])
                j += 1
            is_named = bool(ns_tokens)
            is_alias = any(t.kind == "OP" and t.text == "=" for t in ns_tokens)
            if is_named and (mode == "header" or is_alias):
                ns_name = "".join(t.text for t in ns_tokens if t.kind != "OP" or t.text == "::")
                _validate_doc_block(
                    doc=decl_doc,
                    decl_line=tok.line,
                    rel=rel,
                    kind_label=f"namespace '{ns_name}'",
                    tparams=[],
                    params=[],
                    require_return=False,
                    errors=errors,
                )
            if j < n_tok and tokens[j].kind == "OP" and tokens[j].text == "{":
                scopes.append(Scope("namespace", brace_depth=1))
                idx = j + 1
            else:
                idx = j + 1
            continue

        # Collect header tokens of a general declaration up to `;` or `{` (at depth 0).
        decl_tokens: list[Token] = []
        paren = 0
        bracket = 0
        angle = 0
        end_kind = ""
        while idx < n_tok:
            cur = tokens[idx]
            if cur.kind in {"DOX", "COMMENT", "BLANK", "PREPROC"}:
                idx += 1
                continue
            if cur.kind == "OP":
                if cur.text == "(":
                    paren += 1
                elif cur.text == ")":
                    paren = max(0, paren - 1)
                elif cur.text == "[":
                    bracket += 1
                elif cur.text == "]":
                    bracket = max(0, bracket - 1)
                elif cur.text == "<" and paren == 0 and bracket == 0:
                    angle += 1
                elif cur.text == ">" and paren == 0 and bracket == 0:
                    angle = max(0, angle - 1)
                elif cur.text == ">>" and paren == 0 and bracket == 0 and angle >= 2:
                    angle -= 2
                elif paren == 0 and bracket == 0 and cur.text in {";", "{"}:
                    # Distinguish ctor initializer `{...}` from function body `{` below.
                    end_kind = cur.text
                    idx += 1
                    break
            decl_tokens.append(cur)
            idx += 1

        if not decl_tokens:
            continue

        stripped_decl = _strip_leading_attributes(decl_tokens)
        if not stripped_decl:
            if end_kind == "{":
                scopes.append(Scope("braced_init", brace_depth=1))
            continue

        decl_line = stripped_decl[0].line
        first_kw = stripped_decl[0].text if stripped_decl[0].kind == "IDENT" else ""

        # Check `using`.
        if first_kw == "using":
            if len(stripped_decl) >= 2 and stripped_decl[1].text == "namespace":
                pass
            else:
                alias_name = stripped_decl[1].text if len(stripped_decl) >= 2 else "alias"
                _validate_doc_block(
                    doc=decl_doc,
                    decl_line=decl_line,
                    rel=rel,
                    kind_label=f"using '{alias_name}'",
                    tparams=tparams,
                    params=[],
                    require_return=False,
                    errors=errors,
                )
            if end_kind == "{":
                scopes.append(Scope("braced_init", brace_depth=1))
            continue

        # Check `concept`.
        if first_kw == "concept":
            cname = stripped_decl[1].text if len(stripped_decl) >= 2 else "concept"
            _validate_doc_block(
                doc=decl_doc,
                decl_line=decl_line,
                rel=rel,
                kind_label=f"concept '{cname}'",
                tparams=tparams,
                params=[],
                require_return=False,
                errors=errors,
            )
            if end_kind == "{":
                # Skip `requires { ... };` until terminating `;`.
                bdepth = 1
                while idx < n_tok:
                    if tokens[idx].kind == "OP" and tokens[idx].text == "{":
                        bdepth += 1
                    elif tokens[idx].kind == "OP" and tokens[idx].text == "}":
                        bdepth -= 1
                    elif tokens[idx].kind == "OP" and tokens[idx].text == ";" and bdepth == 0:
                        idx += 1
                        break
                    idx += 1
            continue

        # Check `enum`.
        if first_kw == "enum":
            ename = "enum"
            for t in stripped_decl[1:]:
                if t.kind == "IDENT" and t.text not in {"class", "struct"}:
                    ename = t.text
                elif t.kind == "OP" and t.text == ":":
                    break
            _validate_doc_block(
                doc=decl_doc,
                decl_line=decl_line,
                rel=rel,
                kind_label=f"enum '{ename}'",
                tparams=tparams,
                params=[],
                require_return=False,
                errors=errors,
            )
            if end_kind == "{":
                scopes.append(Scope("enum", name=ename, brace_depth=1))
            continue

        # Check `class` / `struct` / `union` declaration or definition.
        # Skip leading qualifiers like `friend` / `inline`.
        non_qual = [
            t for t in stripped_decl if not (t.kind == "IDENT" and t.text in DECL_SPECIFIER_SKIP)
        ]
        open_paren_idx = _find_top_level_open_paren(stripped_decl)
        has_assign = _has_top_level_assign_before(stripped_decl, open_paren_idx)

        if (
            non_qual
            and non_qual[0].kind == "IDENT"
            and non_qual[0].text in {"class", "struct", "union"}
            and (open_paren_idx == -1 and not has_assign)
        ):
            # Extract class/struct name (last identifier before `final`, `:`, `<`, or end).
            class_name = non_qual[0].text
            for t in non_qual[1:]:
                if t.kind == "IDENT" and t.text != "final":
                    class_name = t.text
                elif t.kind == "OP" and t.text == ":":
                    break
            _validate_doc_block(
                doc=decl_doc,
                decl_line=decl_line,
                rel=rel,
                kind_label=f"{non_qual[0].text} '{class_name}'",
                tparams=tparams,
                params=[],
                require_return=False,
                errors=errors,
            )
            if end_kind == "{":
                scopes.append(Scope("class", name=class_name, brace_depth=1))
            continue

        # Distinguish variable/member field vs function/method/constructor/destructor.
        if open_paren_idx == -1 or has_assign:
            # Variable, constant, or member field.
            var_name = "member"
            limit = len(stripped_decl)
            for k, t in enumerate(stripped_decl):
                if t.kind == "OP" and t.text == "=":
                    limit = k
                    break
            for t in reversed(stripped_decl[:limit]):
                if t.kind == "IDENT":
                    var_name = t.text
                    break
            _validate_doc_block(
                doc=decl_doc,
                decl_line=decl_line,
                rel=rel,
                kind_label=f"member/variable '{var_name}'",
                tparams=tparams,
                params=[],
                require_return=False,
                errors=errors,
            )
            if end_kind == "{":
                scopes.append(Scope("braced_init", brace_depth=1))
            continue

        # Function, method, constructor, destructor, or operator.
        # Handle brace-enclosed member initializers before the function body `{`.
        if end_kind == "{":
            # Detect ctor member initializer list `Ctor(...) : a{1}, b(2) { body }`.
            # Find top-level closing `)` of parameter list inside `stripped_decl`.
            pdepth = 0
            close_paren_idx = -1
            for k in range(open_paren_idx, len(stripped_decl)):
                if stripped_decl[k].kind == "OP" and stripped_decl[k].text == "(":
                    pdepth += 1
                elif stripped_decl[k].kind == "OP" and stripped_decl[k].text == ")":
                    pdepth -= 1
                    if pdepth == 0:
                        close_paren_idx = k
                        break
            has_ctor_colon = False
            if close_paren_idx != -1:
                for t in stripped_decl[close_paren_idx + 1 :]:
                    if t.kind == "OP" and t.text == ":":
                        has_ctor_colon = True
                        break
            if has_ctor_colon:
                # Skip braced initializers inside the ctor init list until the function body `{`.
                # Member initializers follow `member{...}` or `Base<T>{...}` (IDENT or `>`).
                prev_tok = stripped_decl[-1]
                while prev_tok.kind == "IDENT" or (prev_tok.kind == "OP" and prev_tok.text == ">"):
                    # Consume `member{...}` until matching `}`, then advance to the next `{`.
                    bdepth = 1
                    while idx < n_tok and bdepth > 0:
                        if tokens[idx].kind == "OP" and tokens[idx].text == "{":
                            bdepth += 1
                        elif tokens[idx].kind == "OP" and tokens[idx].text == "}":
                            bdepth -= 1
                        idx += 1
                    last_seen = Token("OP", "}", decl_line)
                    while idx < n_tok:
                        c = tokens[idx]
                        if c.kind in {"DOX", "COMMENT", "BLANK", "PREPROC"}:
                            idx += 1
                            continue
                        if c.kind == "OP" and c.text == "{":
                            idx += 1
                            break
                        last_seen = c
                        idx += 1
                    prev_tok = last_seen

        # Extract parameters from `( ... )` starting at `open_paren_idx`.
        pdepth = 0
        close_paren_idx = -1
        for k in range(open_paren_idx, len(stripped_decl)):
            if stripped_decl[k].kind == "OP" and stripped_decl[k].text == "(":
                pdepth += 1
            elif stripped_decl[k].kind == "OP" and stripped_decl[k].text == ")":
                pdepth -= 1
                if pdepth == 0:
                    close_paren_idx = k
                    break

        param_slice = (
            stripped_decl[open_paren_idx + 1 : close_paren_idx] if close_paren_idx != -1 else []
        )
        param_names = _extract_param_names(param_slice)

        # Classify constructor, destructor, conversion operator, or regular function.
        pre_paren = stripped_decl[:open_paren_idx]
        post_paren = stripped_decl[close_paren_idx + 1 :] if close_paren_idx != -1 else []

        is_dtor = any(t.kind == "OP" and t.text == "~" for t in pre_paren)
        is_operator = any(t.kind == "IDENT" and t.text == "operator" for t in pre_paren)

        func_name = "function"
        if is_operator:
            op_idx = next(
                k for k, t in enumerate(pre_paren) if t.kind == "IDENT" and t.text == "operator"
            )
            op_sym = "".join(t.text for t in pre_paren[op_idx + 1 :])
            func_name = f"operator{op_sym or '()'}"
            ret_tokens = [
                t
                for t in pre_paren[:op_idx]
                if not (t.kind == "IDENT" and t.text in DECL_SPECIFIER_SKIP)
            ]
            # Conversion operators (`operator T()`) have empty ret_tokens before `operator`.
            is_conversion_op = len(ret_tokens) == 0 and bool(op_sym) and op_sym[0].isalpha()
            is_ctor = False
        else:
            # Function or constructor/destructor name is the last identifier before `(`.
            func_name = (
                pre_paren[-1].text if pre_paren and pre_paren[-1].kind == "IDENT" else "function"
            )
            # Walk backward past qualified `Scope<Args>::` prefixes to isolate return type tokens.
            enclosing_cls = ""
            name_start = len(pre_paren) - 1
            while (
                name_start >= 2
                and pre_paren[name_start - 1].kind == "OP"
                and pre_paren[name_start - 1].text == "::"
            ):
                q_idx = name_start - 2
                if pre_paren[q_idx].kind == "OP" and pre_paren[q_idx].text in {">", ">>"}:
                    adepth = 2 if pre_paren[q_idx].text == ">>" else 1
                    q_idx -= 1
                    while q_idx >= 0 and adepth > 0:
                        if pre_paren[q_idx].kind == "OP":
                            if pre_paren[q_idx].text == ">":
                                adepth += 1
                            elif pre_paren[q_idx].text == ">>":
                                adepth += 2
                            elif pre_paren[q_idx].text == "<":
                                adepth -= 1
                        q_idx -= 1
                if q_idx >= 0 and pre_paren[q_idx].kind == "IDENT":
                    if not enclosing_cls:
                        enclosing_cls = pre_paren[q_idx].text
                    name_start = q_idx
                else:
                    break
            ret_tokens = [
                t
                for t in pre_paren[:name_start]
                if not (t.kind == "IDENT" and t.text in DECL_SPECIFIER_SKIP)
            ]
            is_ctor = not is_dtor and (
                (cur_scope.kind == "class" and func_name == cur_scope.name and not ret_tokens)
                or (bool(enclosing_cls) and enclosing_cls == func_name and not ret_tokens)
            )
            is_conversion_op = False

        # Check trailing return type `-> ReturnType`.
        trailing_ret_tokens: list[Token] = []
        for k, t in enumerate(post_paren):
            if t.kind == "OP" and t.text == "->":
                for rt in post_paren[k + 1 :]:
                    if rt.kind == "IDENT" and rt.text == "requires":
                        break
                    trailing_ret_tokens.append(rt)
                break

        effective_ret = trailing_ret_tokens or ret_tokens
        returns_void = (
            len(effective_ret) == 1
            and effective_ret[0].kind == "IDENT"
            and effective_ret[0].text == "void"
        )
        require_return = not is_ctor and not is_dtor and (is_conversion_op or not returns_void)

        _validate_doc_block(
            doc=decl_doc,
            decl_line=decl_line,
            rel=rel,
            kind_label=f"function '{func_name}'",
            tparams=tparams,
            params=param_names,
            require_return=require_return,
            errors=errors,
        )

        if end_kind == "{":
            scopes.append(Scope("function_body", brace_depth=1))

    if mode in {"src", "test"} and len(func_body_code_lines) > 25 and inline_comment_count == 0:
        errors.append(
            f"{rel}:1: implementation file has {len(func_body_code_lines)} body lines "
            "but lacks explanatory inline '//' comments"
        )

    return errors


def check_cpp_sources(roots: tuple[tuple[Path, str], ...] = CPP_ROOTS) -> list[str]:
    """Check all C++ headers and sources across include, src, tests, benchmarks, and qa."""
    errors: list[str] = []
    for root, default_mode in roots:
        if not root.exists():
            continue
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".hpp", ".cpp"})
        for file_path in files:
            errors.extend(check_cpp_file(file_path, default_mode))
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


def _run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pub = root / "include" / "cxxgpr"
        src = root / "src"
        pub.mkdir(parents=True)
        src.mkdir(parents=True)

        bad_hpp = pub / "bad.hpp"
        bad_hpp.write_text(
            "// SPDX-License-Identifier: Apache-2.0\n"
            "/// @brief Missing file tag.\n"
            "\n"
            "namespace cxxgpr {\n"
            "/// @brief REQ-01 leak.\n"
            "enum class BadEnum { UndocEnumerator };\n"
            "/// @brief Missing tparam and param and return.\n"
            "template <typename T>\n"
            "int bad_fn(T val);\n"
            "struct UndocStruct { int x; };\n"
            "}\n",
            encoding="utf-8",
        )
        bad_cpp = src / "bad.cpp"
        bad_cpp.write_text(
            "// SPDX-License-Identifier: Apache-2.0\n"
            "/// @file bad.cpp\n"
            "/// @brief Missing inline comments and documented helper.\n"
            "int helper(int a) {\n"
            + "".join("    int v = a;\n" for _ in range(30))
            + "    return a;\n}\n",
            encoding="utf-8",
        )
        good_hpp = pub / "good.hpp"
        good_hpp.write_text(
            "// SPDX-License-Identifier: Apache-2.0\n"
            "/// @file good.hpp\n"
            "/// @brief Well-documented header.\n"
            "#pragma once\n"
            "/// @brief Root namespace.\n"
            "namespace cxxgpr\n"
            "{\n"
            "    /// @brief Sample mode enum.\n"
            "    enum class GoodMode\n"
            "    {\n"
            "        /// @brief First option.\n"
            "        First,\n"
            "    };\n"
            "    /// @brief Sample value struct.\n"
            "    /// @tparam T Value scalar type.\n"
            "    template <typename T>\n"
            "    struct GoodBox\n"
            "    {\n"
            "        /// @brief Stored scalar value.\n"
            "        T value = T{0};\n"
            "        /// @brief Checks equality.\n"
            "        /// @param other Value to compare against.\n"
            "        /// @return True when equal.\n"
            "        bool operator==(const GoodBox& other) const = default;\n"
            "        /// @brief Constructs a GoodBox.\n"
            "        /// @param initial Initial stored value.\n"
            "        GoodBox(T initial);\n"
            "        /// @brief Resets value to zero.\n"
            "        void reset();\n"
            "    };\n"
            "    /// @brief Out-of-line constructor.\n"
            "    /// @tparam T Value scalar type.\n"
            "    /// @param initial Initial stored value.\n"
            "    template <typename T> GoodBox<T>::GoodBox(T initial) : value{initial} {}\n"
            "    /// @brief Out-of-line void method.\n"
            "    /// @tparam T Value scalar type.\n"
            "    template <typename T> void GoodBox<T>::reset() { value = T{0}; }\n"
            "    /// @brief Scales input by two.\n"
            "    /// @tparam T Scalar type.\n"
            "    /// @param x Input scalar.\n"
            "    /// @return Doubled value.\n"
            "    template <typename T> T twice(T x) { return x + x; }\n"
            "}\n",
            encoding="utf-8",
        )

        hpp_errs = check_cpp_file(bad_hpp, "header")
        cpp_errs = check_cpp_file(bad_cpp, "src")
        good_errs = check_cpp_file(good_hpp, "header")
        if len(hpp_errs) < 8:
            print(f"Self-test expected >= 8 header errors, got {len(hpp_errs)}: {hpp_errs}")
            return 1
        if len(cpp_errs) < 2:
            print(f"Self-test expected >= 2 src errors, got {len(cpp_errs)}: {cpp_errs}")
            return 1
        if good_errs:
            print(f"Self-test expected 0 errors for good.hpp, got {good_errs}")
            return 1
    print("Doc check self-test passed.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="Run built-in C++ Doxygen check self-test."
    )
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()

    errors = [*check_cpp_sources(), *check_markdown_links()]
    for error in errors:
        print(error)
    if not errors:
        print("Documentation check passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
