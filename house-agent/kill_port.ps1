$connections = Get-NetTCPConnection -LocalPort 2024 -ErrorAction SilentlyContinue
if ($connections) {
    foreach ($conn in $connections) {
        $targetPid = $conn.OwningProcess
        $procName = (Get-Process -Id $targetPid -ErrorAction SilentlyContinue).ProcessName
        Write-Host "Killing process: PID=$targetPid, Name=$procName, Port=2024"
        Stop-Process -Id $targetPid -Force -ErrorAction Stop
        Write-Host "Process $targetPid killed successfully."
    }
} else {
    Write-Host "No process found on port 2024"
}

# Wait a moment and verify
Start-Sleep -Seconds 1
$remaining = Get-NetTCPConnection -LocalPort 2024 -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "ERROR: Port 2024 is STILL in use by PID $($remaining.OwningProcess)!"
    Write-Host "Try running this script as Administrator."
} else {
    Write-Host "SUCCESS: Port 2024 is now FREE"
}
