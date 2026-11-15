# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$InvocationArgs
)

$ErrorActionPreference = 'Stop'

if (-not $InvocationArgs -or $InvocationArgs.Count -eq 0) {
    throw 'Usage: msvc.ps1 <command> [args...]'
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at '$vswhere'"
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw 'Visual Studio installation with VC x86/x64 tools not found'
}

$vsDevCmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat not found at '$vsDevCmd'"
}

$savedVcpkg = $env:VCPKG_ROOT
if (-not $savedVcpkg -and (Test-Path 'C:\vcpkg\scripts\buildsystems\vcpkg.cmake')) {
    $savedVcpkg = 'C:\vcpkg'
}

$envOutput = cmd.exe /d /s /c "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
}

foreach ($line in $envOutput) {
    $idx = $line.IndexOf('=')
    if ($idx -gt 0) {
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

if ($savedVcpkg) {
    [Environment]::SetEnvironmentVariable('VCPKG_ROOT', $savedVcpkg, 'Process')
}

$extraPaths = @()
$llvmBin = Join-Path $vsPath 'VC\Tools\Llvm\x64\bin'
if (Test-Path $llvmBin) {
    $extraPaths += $llvmBin
}
$cmakeBin = 'C:\Program Files\CMake\bin'
if (Test-Path $cmakeBin) {
    $extraPaths += $cmakeBin
}
if ($extraPaths.Count -gt 0) {
    $env:PATH = ($extraPaths -join ';') + ';' + $env:PATH
}

$exe = $InvocationArgs[0]
if ($InvocationArgs.Count -gt 1) {
    $rest = $InvocationArgs[1..($InvocationArgs.Count - 1)]
    & $exe @rest
} else {
    & $exe
}
exit $LASTEXITCODE
