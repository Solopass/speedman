@echo off
if "%~1"=="" (
    echo Drag and drop an audio file onto this batch file or use Explorer SendTo.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0speedman-sendto.ps1" -InputPath "%~1" -Transcribe -Obsidian
