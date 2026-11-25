$ErrorActionPreference = 'Stop'
$reproRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $reproRoot
try {
    $env:PYTHONPATH = Join-Path $reproRoot 'src'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:MPLBACKEND = 'Agg'
    & python -m pm25_repro all @args
    if ($LASTEXITCODE -ne 0) { throw "Reproduction failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
