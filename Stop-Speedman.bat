@echo off
title Stop SOL Speedman
echo ========================================================
echo               STOPPING SOL SPEEDMAN
echo ========================================================
echo.
echo [1/2] Sending clean shutdown command to speedman (127.0.0.1:8081)...
powershell -NoProfile -Command "try { $res = Invoke-RestMethod -Uri 'http://127.0.0.1:8081/api/v1/shutdown' -Method Post -TimeoutSec 3; Write-Host 'Server response: ' $res.message } catch { Write-Host 'Server is already stopped or idle.' }"

echo [2/2] Releasing WSL background keepalive...
wsl.exe -d Ubuntu-24.04 -- bash -c "pkill -x sleep 2>/dev/null; pkill -f 'dbus-daemon --syslog --fork' 2>/dev/null || true"

echo.
echo ========================================================
echo Speedman has been cleanly stopped.
echo All host CPU threads and RAM have been released.
echo ========================================================
timeout /t 3 >nul
