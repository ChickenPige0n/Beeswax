param([string]$RuntimeDll = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeDll) {
    $RuntimeDll = & python -c "import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).parent / 'capi' / 'onnxruntime.dll')"
    if ($LASTEXITCODE -ne 0) { throw 'Install the CPU onnxruntime Python package, or pass -RuntimeDll with an ONNX Runtime DLL path.' }
}
if (-not (Test-Path -LiteralPath $RuntimeDll -PathType Leaf)) { throw "Runtime DLL not found: $RuntimeDll" }
$runtimeDirectory = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
Copy-Item -LiteralPath $RuntimeDll -Destination (Join-Path $runtimeDirectory 'onnxruntime.dll')
# ONNX Runtime loads providers from beside onnxruntime.dll. A GPU package ships
# onnxruntime_providers_cuda.dll there, which is what makes the CUDA backend selectable.
Get-ChildItem -LiteralPath (Split-Path -Parent $RuntimeDll) -Filter 'onnxruntime_providers_*.dll' |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $runtimeDirectory -Force }
if (-not (Test-Path -LiteralPath (Join-Path $runtimeDirectory 'onnxruntime_providers_cuda.dll'))) {
    Write-Output 'No CUDA provider DLL found; the CPU backend is the only option with this runtime.'
}
Write-Output "Runtime copied to $runtimeDirectory"
