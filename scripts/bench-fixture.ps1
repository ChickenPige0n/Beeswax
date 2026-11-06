param(
    [string]$Output = ""
)

# Generate the deterministic competing-talker clip used by src/bin/bench.rs.
# The same voice, rate, text and PCM format produce identical bytes on every
# run, so quality numbers stay comparable between benchmark runs.

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($Output)) {
    $Output = Join-Path $root "target/autoresearch/interference.wav"
}

$text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. She sells sea shells by the sea shore, and the shells she sells are surely seashells."
$voiceName = "Microsoft Zira Desktop"

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$installed = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
if ($installed -notcontains $voiceName) {
    $synth.Dispose()
    throw "The benchmark fixture needs the $voiceName voice. Installed voices: $($installed -join ', ')"
}
$synth.SelectVoice($voiceName)
$synth.Rate = 0

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
$synth.SetOutputToWaveFile($Output, $format)
$synth.Speak($text)
$synth.SetOutputToNull()
$synth.Dispose()

Write-Output "Wrote $Output"
