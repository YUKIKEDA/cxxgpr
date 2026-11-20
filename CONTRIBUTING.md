# Development workflow

The entry point for agents is [AGENTS.md](AGENTS.md), and the rules are in `.cursor/rules/`. This document summarizes the development workflow.

## Order

Grill (when the design forks) → Issue → branch → work → PR → review → squash merge → retrospective. CI runs asynchronously on the PR, and the successes required for merge are confirmed before merging.

- Create Issues from the templates in `.github/ISSUE_TEMPLATE/`. Do not create blank Issues. Write the row ID from [docs/internal/roadmap.md](docs/internal/roadmap.md) in the Issue.
- Branch names are `type/<issue-number>-<slug>`. Never commit directly to `main`.
- One Issue is one PR. The owner performs the merge and deletes the merged branch.
- A requested task proceeds through checks, commit, a normal push, and PR creation. If the design or scope decision forks, confirm before the PR.
- After creating a PR, an agent does not in principle wait for CI to finish; it reports the PR URL and the run status and returns.
- Commit subjects are `type(scope): subject` (English, imperative mood, no trailing period).

| Template | Label | Commit type |
|---|---|---|
| feat | `type:feat` | `feat` |
| bug | `type:bug` | `fix` |
| refactor | `type:refactor` | `refactor` |
| test | `type:test` | `test` |
| design | `type:docs` | `docs` |
| task | `type:chore` | `chore` `build` `ci` |
| spike | `type:spike` | `chore` |

## Retrospective

After the merge, write three points on the PR: the lessons learned, the file to write them into, and whether additional refactoring is needed. If there is a lesson, open a `chore` Issue and write it into the rules, Skills, specifications, or ADRs.

## Gates

Choose the checks that affect the diff and its dependents. Use local tests during work, and consolidate any full verification into one run at the very end after editing is finished. The gate suite is implemented in `justfile` and CI in M0-2 (Issue #2).

| Command | Contents | Gate |
|---|---|---|
| `just gate` | Configure, build, and test (Windows Debug/Release, Linux Debug/Release/Sanitize) | yes |
| `just format-check` | No `clang-format` differences | yes |
| `just lint` | No `clang-tidy` findings (WarningsAsErrors) | yes |
| `just lint-python` | Python tools: `ruff` (check and format) and `pyright` strict | yes |
| `just trace-check` | Consistency of specification IDs (`REQ-`/`DES-`/`FMT-`/`QA-`) with the traceability matrix | yes |
| `just arch-check` | Dependency direction and non-exposure of third-party headers in `include/cxxgpr/` | yes |
| `just doc-check` | Presence of public Doxygen comments and absence of internal IDs | yes |
| `just skills-check` | `.claude/skills` is an exact copy of `.agents/skills` | yes |
| `just results-check` | Agreement with the baseline results (`qa/baselines/`) | yes |
| `just format` / `just skills-sync` | Formatting; Skill copying | no |

## Working in parallel

Before starting, sort out each task's prerequisites, deliverables, files to edit, and shared state. Do dependent work in order, and run in parallel the work that needs none of the others' results and does not collide on shared files. Do not edit a file while building or analyzing it at the same time.

## Language

- **User documentation** (`docs/en/`, `docs/ja/`, `README.md`, `README.ja.md`) is bilingual: English and Japanese. Keep both languages.
- **Developer documentation** (`docs/internal/`, `CONTRIBUTING.md`) is English only.
- **Agent configuration** (`.cursor/rules/`, `AGENTS.md`, `CLAUDE.md`, Skills) is written in Japanese.
- Issue and PR bodies are written in Japanese (plain form, joyo style). Code, Doxygen comments, identifiers, commit subjects, and PR titles are English.
