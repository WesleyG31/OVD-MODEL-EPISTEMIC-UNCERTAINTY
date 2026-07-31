param(
    [ValidateSet("gpu", "cpu")]
    [string]$Target = "gpu",
    [string]$Python = "py",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentPath = Join-Path $ProjectRoot ".venv"

if ($Recreate -and (Test-Path -LiteralPath $EnvironmentPath)) {
    $ResolvedEnvironment = (Resolve-Path -LiteralPath $EnvironmentPath).Path
    $ResolvedProject = (Resolve-Path -LiteralPath $ProjectRoot).Path
    if (-not $ResolvedEnvironment.StartsWith($ResolvedProject)) {
        throw "Refusing to remove an environment outside the project."
    }
    Remove-Item -LiteralPath $ResolvedEnvironment -Recurse -Force
}

if (-not (Test-Path -LiteralPath $EnvironmentPath)) {
    if ($Python -eq "py") {
        & py -3.12 -m venv $EnvironmentPath
    } else {
        & $Python -m venv $EnvironmentPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment with $Python."
    }
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
if ($Target -eq "gpu" -and -not $env:CUDA_HOME) {
    $CudaCandidate = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    if (Test-Path -LiteralPath $CudaCandidate) {
        $env:CUDA_HOME = $CudaCandidate
    }
}
& $EnvironmentPython -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
if ($LASTEXITCODE -ne 0) { throw "Failed to install packaging tools." }
if ($Target -eq "gpu") {
    & $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot "requirements-gpu-cu118.txt")
} else {
    & $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot "requirements-cpu.txt")
}
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch." }
& $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot "requirements-common.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies." }

& $EnvironmentPython -m pip install --no-deps -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install the research package." }
& $EnvironmentPython -m pytest `
    (Join-Path $ProjectRoot "tests") `
    (Join-Path $ProjectRoot "RQ1\tests") `
    (Join-Path $ProjectRoot "RQ2\tests") `
    (Join-Path $ProjectRoot "RQ3\tests") `
    (Join-Path $ProjectRoot "RQ4\tests") `
    (Join-Path $ProjectRoot "RQ5\tests") `
    --basetemp (Join-Path $ProjectRoot ".pytest-tmp") -q
if ($LASTEXITCODE -ne 0) { throw "Project tests failed." }
if ($Target -eq "gpu") {
    & $EnvironmentPython (Join-Path $ProjectRoot "scripts\verify_environment.py")
} else {
    & $EnvironmentPython (Join-Path $ProjectRoot "scripts\verify_environment.py") --allow-cpu
}
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }
& $EnvironmentPython -m pip freeze --exclude-editable |
    Set-Content -Encoding UTF8 (Join-Path $ProjectRoot "requirements-lock.txt")
