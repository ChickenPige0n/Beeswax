param(
    [ValidateSet('fp32', 'fp16')][string]$Precision = 'fp32',
    [string]$BaseUrl = 'https://huggingface.co'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$models = Join-Path $projectRoot 'models'
$revision = 'a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03'
New-Item -ItemType Directory -Path $models -Force | Out-Null

# The separation model ships in int8 only. The fp32/fp16 variants sound cleaner but
# need more compute; pair them with the CUDA backend on a CPU that cannot keep up.
$file = if ($Precision -eq 'fp16') { 'onnx/chunk_fp16.onnx' } else { 'onnx/chunk.onnx' }
$destination = Join-Path $models (Split-Path -Leaf $file)
$url = "$BaseUrl/arbazsiddiqui/voxlock/resolve/$revision/$file"
$expectedHash = if ($Precision -eq 'fp16') {
    'd6e931ea261d217d03bdad05aa8e1d6a11a41d4a421a90ea91622d8ec17f8e82'
} else {
    '3c31444926cdf3d548935344e2f5f1c71538ded6c2c7d1bd22e2c2141ea0bc2a'
}
$partial = "$destination.download"

Write-Output "Downloading $url"
$null = Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
$actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    Remove-Item -LiteralPath $partial -Force
    throw 'Downloaded model checksum does not match the VoxLock 1.1.1 model revision.'
}
Move-Item -LiteralPath $partial -Destination $destination -Force
Get-Item -LiteralPath $destination | Select-Object Name, Length

if ($Precision -eq 'fp16') {
    Write-Output 'fp16 is only faster when the CUDA backend is used.'
}
Write-Output 'Beeswax prefers fp32 > fp16 > int8 automatically on the next Start filtering.'
