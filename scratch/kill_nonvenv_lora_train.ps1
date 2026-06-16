# 结束「非本仓库 venv」发起的 V5A1/V5B1 LoRA 训练进程，保留 `...\venv\Scripts\python.exe` 那一套。
# 适用：Cursor/任务计划误用系统 Python 导致与 venv 双开抢写 OUTPUT_DIR。
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_v5a1\.py|train_v5b1\.py' } |
    Where-Object { $_.CommandLine -notmatch '[\\/]venv[\\/]Scripts[\\/]python\.exe' } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $($_.ProcessId)"
    }

Write-Host "Done. Remaining LoRA train processes (if any):"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_v5a1\.py|train_v5b1\.py' } |
    ForEach-Object { "  PID $($_.ProcessId) $($_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))" }
