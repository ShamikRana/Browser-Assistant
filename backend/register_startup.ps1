<#
Registers a Windows Scheduled Task that starts the Browser Assistant (Phi-4)
backend automatically when you log in, so it is already running (and warmed up)
by the time you open the browser.

The task runs pythonw.exe directly - no console window, no interactive prompts -
and restarts automatically if the backend ever exits unexpectedly.

Usage:
    .\register_startup.ps1                  # auto-detect device
    .\register_startup.ps1 -Device gpu
    .\register_startup.ps1 -StartNow        # also start it immediately
    .\unregister_startup.ps1                # remove it
#>
param(
    [ValidateSet("auto", "cpu", "gpu")]
    [string]$Device = "auto",

    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$TaskName = "BrowserAssistantPhi4"
$ScriptDir = $PSScriptRoot
$ServePath = Join-Path $ScriptDir "serve.py"

# Prefer the project virtual environment; pythonw.exe runs without a console window.
$VenvPythonw = Join-Path $ScriptDir "..\venv\Scripts\pythonw.exe"
if (Test-Path $VenvPythonw) {
    $Python = (Resolve-Path $VenvPythonw).Path
} else {
    $Command = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $Command) {
        throw "pythonw.exe not found. Create the venv first, or install Python and retry."
    }
    $Python = $Command.Source
    Write-Warning "Project venv not found; using system Python at $Python"
}

$Arguments = "`"$ServePath`" --device $Device --log-file logs\server.log"

$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $ScriptDir

$Trigger = New-ScheduledTaskTrigger -AtLogOn
# Let the desktop finish loading before the model starts loading.
$Trigger.Delay = "PT30S"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$Settings.Hidden = $true

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings `
    -Description "Starts the Browser Assistant (Phi-4) local backend at logon" -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName' (device: $Device)."
Write-Host "Logs: $(Join-Path $ScriptDir 'logs\server.log')"

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Task started. The backend will be ready once the model finishes loading."
}

Write-Host "Run '.\unregister_startup.ps1' to remove it."
