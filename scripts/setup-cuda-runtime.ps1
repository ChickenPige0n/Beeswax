param(
    [string]$OrtWheel = '',
    [string]$OrtDll = '',
    [string]$CudaDllDir = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

# Keep the CPU runtime around so a failed GPU experiment can be reverted by renaming it back.
$current = Join-Path $runtime 'onnxruntime.dll'
if ((Test-Path -LiteralPath $current) -and -not (Test-Path -LiteralPath (Join-Path $runtime 'onnxruntime-cpu.dll'))) {
    Copy-Item -LiteralPath $current -Destination (Join-Path $runtime 'onnxruntime-cpu.dll')
    Write-Output 'Backed up the previous runtime as runtime\onnxruntime-cpu.dll'
}

if ($OrtWheel) {
    if (-not (Test-Path -LiteralPath $OrtWheel)) { throw "Wheel not found: $OrtWheel" }
    $extract = Join-Path $env:TEMP ('beeswax-ort-' + [guid]::NewGuid().ToString('N'))
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory((Resolve-Path -LiteralPath $OrtWheel), $extract)
    $capi = Join-Path $extract 'onnxruntime/capi'
    if (-not (Test-Path -LiteralPath (Join-Path $capi 'onnxruntime_providers_cuda.dll'))) {
        throw 'This wheel has no onnxruntime_providers_cuda.dll. Download onnxruntime-gpu, not onnxruntime.'
    }
    Copy-Item -LiteralPath (Join-Path $capi 'onnxruntime.dll') -Destination $runtime -Force
    Get-ChildItem -LiteralPath $capi -Filter 'onnxruntime_providers_*.dll' |
        Copy-Item -Destination $runtime -Force
    Remove-Item -LiteralPath $extract -Recurse -Force
    Write-Output "CUDA ONNX Runtime copied into $runtime"
} elseif ($OrtDll) {
    $source = Split-Path -Parent (Resolve-Path -LiteralPath $OrtDll)
    if (-not (Test-Path -LiteralPath (Join-Path $source 'onnxruntime_providers_cuda.dll'))) {
        throw 'onnxruntime_providers_cuda.dll is missing next to that DLL. Use the onnxruntime-gpu build.'
    }
    Copy-Item -LiteralPath $OrtDll -Destination $runtime -Force
    Get-ChildItem -LiteralPath $source -Filter 'onnxruntime_providers_*.dll' |
        Copy-Item -Destination $runtime -Force
    Write-Output "CUDA ONNX Runtime copied into $runtime"
} else {
    Write-Output 'No runtime given. Download the GPU wheel first, for example:'
    Write-Output '  python -m pip download --no-deps --dest .\target\ort-gpu onnxruntime-gpu==1.30.0'
    Write-Output '  powershell -ExecutionPolicy Bypass -File scripts/setup-cuda-runtime.ps1 -OrtWheel .\target\ort-gpu\onnxruntime_gpu-1.30.0-*.whl'
    throw 'Nothing installed.'
}

# ONNX Runtime's CUDA provider needs CUDA 12.x and cuDNN 9 DLLs, which the wheel does
# not bundle. PyTorch's CUDA build ships exactly these; point the app at them.
if (-not $CudaDllDir) {
    $torchLib = & python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $torchLib -and (Test-Path -LiteralPath $torchLib)) { $CudaDllDir = $torchLib }
}
if ($CudaDllDir) {
    $resolved = (Resolve-Path -LiteralPath $CudaDllDir).Path
    $needed = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll', 'cufft64_11.dll', 'cudnn64_9.dll')
    $missing = $needed | Where-Object { -not (Test-Path -LiteralPath (Join-Path $resolved $_)) }
    if ($missing) {
        Write-Output ("Missing from " + $resolved + ": " + ($missing -join ', '))
    }
    Set-Content -LiteralPath (Join-Path $runtime 'cuda-path.txt') -Value $resolved -Encoding ascii
    Write-Output "CUDA/cuDNN DLLs will be preloaded from $resolved"
} else {
    Write-Output 'No CUDA 12 / cuDNN 9 DLL directory found. Pass -CudaDllDir or set BEESWAX_CUDA_DLLS.'
}

Write-Output 'Check with: .\target\release\beeswax.exe --providers'
