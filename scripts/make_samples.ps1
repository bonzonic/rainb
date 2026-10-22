# Generates the "Play a sample conversation" clips for the demo persona with Windows TTS.
# 16 kHz mono 16-bit, the same format the phone streams. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\make_samples.ps1
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot "..\web\samples"
New-Item -ItemType Directory -Force $out | Out-Null
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$samples = @{
  "vector-db" = @"
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
I had coffee with Daniel last week. <break time="400ms"/>
He told me about the vector database they used at Grab. <break time="300ms"/>
It was called <break time="350ms"/> uh <break time="2500ms"/>
Yeah, <break time="200ms"/> that one. We should try it for Kestrel.
<break time="1500ms"/>
</speak>
"@
  "paper" = @"
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
For the memo to Hendrik, <break time="300ms"/> we should cite that paper, <break time="300ms"/>
the one about retrieval augmented generation. <break time="300ms"/> It's by <break time="2500ms"/>
Right. <break time="300ms"/> And it came out in twenty twenty.
<break time="1500ms"/>
</speak>
"@
  "ramen" = @"
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
Let's take Farid to lunch on Friday. <break time="400ms"/>
How about the ramen place near the office, <break time="300ms"/> the one on Jalan Telawi, <break time="300ms"/> uh, <break time="300ms"/> what's it called <break time="2500ms"/>
Yes! Get the yuzu shio.
<break time="1500ms"/>
</speak>
"@
}

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like "en-*" -and $_.VoiceInfo.Gender -eq "Female" } | Select-Object -First 1
if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) }
$synth.Rate = 0
foreach ($k in $samples.Keys) {
  $path = Join-Path $out "$k.wav"
  $synth.SetOutputToWaveFile($path, $fmt)
  $synth.SpeakSsml($samples[$k])
  $synth.SetOutputToNull()
  Write-Output "wrote $path"
}
$synth.Dispose()
