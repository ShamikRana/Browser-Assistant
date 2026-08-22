<#
Removes the scheduled task created by register_startup.ps1.

Usage:
    .\unregister_startup.ps1
#>

$TaskName = "BrowserAssistantPhi4"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
} else {
    Write-Host "Scheduled task '$TaskName' is not registered."
}
