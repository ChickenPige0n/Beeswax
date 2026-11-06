$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$binary = Join-Path $PSScriptRoot 'target\release\beeswax.exe'
if (-not (Test-Path -LiteralPath $binary)) {
    cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Beeswax build failed.' }
}
& $binary
