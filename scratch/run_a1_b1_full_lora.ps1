# Sequential full-epoch LoRA: V5A1 (MolT5) completes all epochs, then V5B1 (ReactionT5).
# Requires: repo root = working directory; venv present.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

Set-Location -LiteralPath "e:\DA\DeepMet\SC_CLM"
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$py = Join-Path (Get-Location) "venv\Scripts\python.exe"
$chainTranscript = Join-Path (Get-Location) "logs\lora_chain_$ts.transcript.log"
$planFile = Join-Path (Get-Location) "_lora_chain_status.txt"

New-Item -ItemType Directory -Force -Path (Split-Path $chainTranscript) | Out-Null

function Write-Plan([string]$Line) {
    Add-Content -LiteralPath $planFile -Value "$(Get-Date -Format 'o') $Line" -Encoding UTF8
}

Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
Start-Transcript -LiteralPath $chainTranscript -Force | Out-Null

Write-Host "[chain $ts] Transcript -> $chainTranscript"
Write-Plan "PLAN ts=$ts : Phase A V5A1 full epochs -> Phase B V5B1 full epochs"

Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'train_v5a1\.py|train_v5b1\.py'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$env:PYTHONFAULTHANDLER = "1"

# ----- Phase A: MolT5 + LoRA (wrapper sets family=molt5 inside Python) -----
$env:V5B1_MAX_STEPS = "-1"
$env:V5B1_MODEL_FAMILY = "molt5"
$env:V5B1_FORCE_FP16 = "0"
Remove-Item Env:V5B1_NUM_EPOCHS -ErrorAction SilentlyContinue
$env:V5B1_OUTPUT_DIR = "results/checkpoints/v5a1_lora_full_$ts"
$env:V5B1_LOG_FILE = "logs/v5a1/train_lora_full_$ts.log"

Write-Host "[chain] PHASE A V5A1 -> $($env:V5B1_OUTPUT_DIR)"
Write-Plan "START A1 out=$($env:V5B1_OUTPUT_DIR) log=$($env:V5B1_LOG_FILE)"

& $py src/model/v5/train_v5a1.py
$a1 = $LASTEXITCODE
Write-Plan "END A1 exit=$a1"
if ($a1 -ne 0) {
    Stop-Transcript | Out-Null
    exit $a1
}

# ----- Phase B: ReactionT5 + LoRA -----
$env:V5B1_MAX_STEPS = "-1"
$env:V5B1_MODEL_FAMILY = "reactiont5"
$env:V5B1_FORCE_FP16 = "1"
Remove-Item Env:V5B1_NUM_EPOCHS -ErrorAction SilentlyContinue
$env:V5B1_OUTPUT_DIR = "results/checkpoints/v5b1_lora_full_$ts"
$env:V5B1_LOG_FILE = "logs/v5b1/train_lora_full_$ts.log"

Write-Host "[chain] PHASE B V5B1 -> $($env:V5B1_OUTPUT_DIR)"
Write-Plan "START B1 out=$($env:V5B1_OUTPUT_DIR) log=$($env:V5B1_LOG_FILE)"

& $py src/model/v5/train_v5b1.py
$b1 = $LASTEXITCODE
Write-Plan "END B1 exit=$b1"

Stop-Transcript | Out-Null
exit $b1
