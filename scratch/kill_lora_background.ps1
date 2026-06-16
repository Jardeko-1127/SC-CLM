# Stop SC-CLM LoRA chain runners and Python trainers (project-local patterns).
Set-Location -LiteralPath "e:\DA\DeepMet\SC_CLM"
$patterns = @('train_v5a1\.py', 'train_v5b1\.py', 'run_a1_b1_full_lora\.ps1')
Get-CimInstance Win32_Process | ForEach-Object {
    $cmd = $_.CommandLine
    if (-not $cmd) { return }
    foreach ($p in $patterns) {
        if ($cmd -match $p) {
            Write-Host "Stopping PID=$($_.ProcessId) $($_.Name)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            break
        }
    }
}
Write-Host "kill_lora_background: done."
