# SPDX-License-Identifier: Apache-2.0
# Build (and optionally run) docs/examples against an installed cxxgpr package, like a user project.
#
#   tools/dev/build_examples.ps1                     # install the library, build, run every example
#   tools/dev/build_examples.ps1 -Name first_regression -SkipInstall -Tag mine   # private build dir
param(
    [string]$Name = '',
    [switch]$SkipInstall,
    [string]$Config = 'Release',
    [string]$Tag = ''
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$msvc = Join-Path $PSScriptRoot 'msvc.ps1'
$buildDir = Join-Path $root 'out\build\windows-msvc'
$prefix = Join-Path $root 'out\docs\prefix'
$exBuild = Join-Path $root ('out\docs\examples-build' + $(if ($Tag) { "-$Tag" } else { '' }))

function Invoke-Msvc {
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $msvc @args
    if ($LASTEXITCODE -ne 0) { throw "failed: $args" }
}

if (-not $SkipInstall) {
    Invoke-Msvc cmake --preset windows-msvc
    Invoke-Msvc cmake --build $buildDir --config $Config --target cxxgpr
    Invoke-Msvc cmake --install $buildDir --config $Config --prefix $prefix
}

$deps = Join-Path $buildDir 'vcpkg_installed\x64-windows'
# Ensure OpenBLAS / dependency DLLs are discoverable when running standalone example executables on Windows.
$depsBin = if ($Config -eq 'Debug') { Join-Path $deps 'debug\bin' } else { Join-Path $deps 'bin' }
if (Test-Path $depsBin) {
    $env:PATH = "$depsBin;$env:PATH"
}
# Passed through the environment: a "-D...=C:" argument is parsed as a named parameter by pwsh -File.
$env:CMAKE_PREFIX_PATH = "$prefix;$deps"
Invoke-Msvc cmake -S (Join-Path $root 'docs\examples') -B $exBuild -G 'Ninja Multi-Config' `
    -DCMAKE_CXX_COMPILER=cl
if ($Name) {
    Invoke-Msvc cmake --build $exBuild --config $Config --target "cxxgpr_example_$Name"
    & (Join-Path $exBuild "$Config\$Name.exe")
    exit $LASTEXITCODE
}
Invoke-Msvc cmake --build $exBuild --config $Config
Invoke-Msvc ctest --test-dir $exBuild -C $Config --output-on-failure
