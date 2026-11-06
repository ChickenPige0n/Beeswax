param(
    [ValidateSet('fp32', 'fp16')][string]$Precision = 'fp32',
    [string]$BaseUrl = 'https://huggingface.co'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$models = Join-Path $projectRoot 'models'
New-Item -ItemType Directory -Path $models -Force | Out-Null

# The separation model ships in int8 only. The fp32/fp16 variants sound cleaner but
# need more compute; pair them with the CUDA backend on a CPU that cannot keep up.
$file = if ($Precision -eq 'fp16') { 'onnx/chunk_fp16.onnx' } else { 'onnx/chunk.onnx' }
$destination = Join-Path $models (Split-Path -Leaf $file)
$url = "$BaseUrl/arbazsiddiqui/voxlock/resolve/main/$file"

Write-Output "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $destination -UseBasicParsing
Get-Item -LiteralPath $destination | Select-Object Name, Length

if ($Precision -eq 'fp16') {
    Write-Output 'fp16 is only faster when the CUDA backend is used.'
}
Write-Output 'Beeswax prefers fp32 > fp16 > int8 automatically on the next Start filtering.'
