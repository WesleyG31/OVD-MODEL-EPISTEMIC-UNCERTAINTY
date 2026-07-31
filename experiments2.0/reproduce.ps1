param(
    [ValidateSet("smoke", "mini", "full")]
    [string]$Mode = "smoke",
    [string]$Python = "py",
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $SkipSetup) {
    & (Join-Path $ProjectRoot "setup_env.ps1") `
        -Target gpu `
        -Python $Python
    if ($LASTEXITCODE -ne 0) { throw "Environment setup failed." }
}

$EnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    throw "The project venv is missing. Run without -SkipSetup first."
}

& $EnvironmentPython (Join-Path $ProjectRoot "scripts\prepare_data.py")
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\prepare_model.py")
if ($LASTEXITCODE -ne 0) { throw "Model preparation failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\verify_model.py")
if ($LASTEXITCODE -ne 0) { throw "Pinned model verification failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\run_rq1.py") --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "RQ1 execution failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\run_rq2.py") --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "RQ2 execution failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\run_rq3.py") --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "RQ3 execution failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\run_rq4.py") --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "RQ4 execution failed." }
& $EnvironmentPython (Join-Path $ProjectRoot "scripts\run_rq5.py") --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "RQ5 execution failed." }
