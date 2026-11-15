# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('format-check', 'format', 'lint')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'

function Get-CppFiles {
    param([string[]]$Roots)
    $collected = @()
    foreach ($root in $Roots) {
        if (Test-Path $root) {
            $collected += Get-ChildItem -Path $root -Recurse -File -Include *.hpp, *.cpp |
                Select-Object -ExpandProperty FullName
        }
    }
    return $collected
}

if ($Mode -eq 'format-check') {
    $files = Get-CppFiles -Roots @('include', 'src', 'tests', 'qa', 'benchmarks')
    if ($files.Count -eq 0) {
        exit 0
    }
    clang-format --dry-run --Werror @files
    exit $LASTEXITCODE
}

if ($Mode -eq 'format') {
    $files = Get-CppFiles -Roots @('include', 'src', 'tests', 'qa', 'benchmarks')
    if ($files.Count -gt 0) {
        clang-format -i @files
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    if (Get-Command ruff -ErrorAction SilentlyContinue) {
        ruff check --fix tools
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        ruff format tools
        exit $LASTEXITCODE
    }
    uvx --offline ruff check --fix tools
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uvx --offline ruff format tools
    exit $LASTEXITCODE
}

if ($Mode -eq 'lint') {
    $cc = 'out/build/windows-msvc/compile_commands.json'
    if (-not (Test-Path $cc)) {
        cmake --preset windows-msvc
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    $sources = Get-ChildItem -Path @('src', 'tests', 'qa') -Recurse -File -Filter *.cpp |
        Select-Object -ExpandProperty FullName
    if ($sources.Count -eq 0) {
        exit 0
    }
    clang-tidy -p out/build/windows-msvc --quiet --extra-arg=-Wno-unused-command-line-argument @sources
    exit $LASTEXITCODE
}
