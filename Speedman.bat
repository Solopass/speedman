@echo off
title SOL Speedman Speech Studio
echo ========================================================
echo               SOL SPEEDMAN SPEECH STUDIO
echo ========================================================
echo.
echo Speedman is running at http://127.0.0.1:8081/
echo.
echo * Initializing WSL session keepalive...
wsl.exe -d Ubuntu-24.04 --exec dbus-launch true
echo * Opening browser to Speedman Speech Studio...
echo * Pinned to workstation E-cores (8-15) with Zero-RAM standby.
echo * Close browser tab/window to automatically release RAM after 60s idle.
echo.
start http://127.0.0.1:8081/
wsl.exe -d Ubuntu-24.04 -- sleep infinity
