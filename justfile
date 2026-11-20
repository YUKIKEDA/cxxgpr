set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

msvc_cmd := "powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools/dev/msvc.ps1"

# Configure, build, and run tests for the current OS.
[windows]
gate:
    {{msvc_cmd}} cmake --preset windows-msvc
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    {{msvc_cmd}} cmake --build --preset windows-debug
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    {{msvc_cmd}} cmake --build --preset windows-release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    {{msvc_cmd}} ctest --preset windows-debug
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    {{msvc_cmd}} ctest --preset windows-release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -X utf8 tools/dev/ci_scope_test.py

[unix]
gate:
    cmake --preset linux-debug
    cmake --build --preset linux-debug
    ctest --preset linux-debug
    cmake --preset linux-release
    cmake --build --preset linux-release
    ctest --preset linux-release
    cmake --preset linux-sanitize
    cmake --build --preset linux-sanitize
    ctest --preset linux-sanitize
    python3 -X utf8 tools/dev/ci_scope_test.py

# Check C++ formatting with clang-format.
[windows]
format-check:
    {{msvc_cmd}} powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools/dev/cpp_checks.ps1 format-check

[unix]
format-check:
    find include src tests qa benchmarks -type f \( -name '*.hpp' -o -name '*.cpp' \) | xargs -r clang-format --dry-run --Werror

# Format C++ and Python sources.
[windows]
format:
    {{msvc_cmd}} powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools/dev/cpp_checks.ps1 format

[unix]
format:
    find include src tests qa benchmarks -type f \( -name '*.hpp' -o -name '*.cpp' \) | xargs -r clang-format -i
    uvx ruff check --fix tools
    uvx ruff format tools

# Run clang-tidy on compiled C++ translation units.
[windows]
lint:
    {{msvc_cmd}} powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools/dev/cpp_checks.ps1 lint

[unix]
lint:
    test -f out/build/linux-debug/compile_commands.json || cmake --preset linux-debug
    find src tests qa -type f -name '*.cpp' | xargs -r clang-tidy -p out/build/linux-debug

# Lint and type-check Python tools (ruff + pyright strict).
[windows]
lint-python:
    if (Get-Command ruff -ErrorAction SilentlyContinue) { ruff check tools; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; ruff format --check tools; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } else { uvx --offline ruff check tools; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; uvx --offline ruff format --check tools; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
    if (Get-Command pyright -ErrorAction SilentlyContinue) { pyright; exit $LASTEXITCODE } else { uvx --offline pyright; exit $LASTEXITCODE }

[unix]
lint-python:
    uvx ruff check tools
    uvx ruff format --check tools
    uvx pyright

# Check skills mirror consistency.
skills-check:
    python -X utf8 tools/dev/sync_skills.py --check

# Mirror .agents/skills to .claude/skills.
skills-sync:
    python -X utf8 tools/dev/sync_skills.py

# Verify specification traceability and freshness.
trace-check:
    python -X utf8 tools/qa/trace.py check

# Regenerate docs/internal/quality/traceability.md.
trace-matrix:
    python -X utf8 tools/qa/trace.py matrix

# Verify module architecture layering and third-party header isolation.
arch-check:
    python -X utf8 tools/qa/arch_check.py

# Verify English Doxygen coverage, zero internal ID leaks, and markdown links.
doc-check:
    python -X utf8 tools/qa/doc_check.py

# Verify numeric canary matches the platform baseline bit for bit.
[windows]
results-check:
    python -X utf8 tools/qa/results.py check --case numeric-canary --run out/build/windows-msvc/Release/cxxgpr_numeric_canary.exe

[unix]
results-check:
    python3 -X utf8 tools/qa/results.py check --case numeric-canary --run out/build/linux-release/cxxgpr_numeric_canary

# Generate Doxygen XML for the public headers (input of the API reference and the coverage check).
docs-xml:
    python -X utf8 -c "import pathlib; pathlib.Path('out/docs/doxygen').mkdir(parents=True, exist_ok=True)"
    doxygen docs/Doxyfile

# Regenerate the Breathe-backed API reference pages.
docs-api:
    python -X utf8 tools/docs/gen_api_reference.py

# Build the English and Japanese documentation sites with warnings as errors.
[windows]
docs: docs-xml docs-api
    out/docs-venv/Scripts/python.exe -m sphinx -b html -W --keep-going -c docs docs/en out/docs/html/en
    out/docs-venv/Scripts/python.exe -m sphinx -b html -W --keep-going -c docs -D language=ja docs/ja out/docs/html/ja

[unix]
docs: docs-xml docs-api
    python3 -m sphinx -b html -W --keep-going -c docs docs/en out/docs/html/en
    python3 -m sphinx -b html -W --keep-going -c docs -D language=ja docs/ja out/docs/html/ja

# Verify every public API symbol is covered by the user documentation in both languages.
docs-coverage: docs-xml
    python -X utf8 tools/qa/doc_api_coverage.py

# Build and run the documentation example programs against an installed package.
[windows]
docs-examples:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools/dev/build_examples.ps1
