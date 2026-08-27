@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo.
echo  ============================================
echo    speedman setup
echo  ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [X] Python was not found.
    echo.
    echo      Install Python 3.10 or newer from https://python.org/downloads
    echo      IMPORTANT: tick "Add python.exe to PATH" in the installer,
    echo      then run this file again.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v

if not exist ".venv\Scripts\python.exe" (
    echo  [..] Creating a private environment for speedman...
    python -m venv .venv
    if errorlevel 1 (
        echo  [X] Could not create the environment.
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"
echo  [OK] Environment ready

echo  [..] Installing libraries. First run takes a few minutes...
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -e ".[dev]" --quiet
if errorlevel 1 (
    echo  [X] Install failed. Scroll up for the reason.
    pause
    exit /b 1
)
echo  [OK] speedman installed

if not exist "bin\rubberband.exe" (
    echo.
    echo  Rubber Band is the audio engine speedman needs. It is separate
    echo  open-source software ^(GPL^) from breakfastquay.com and is not
    echo  bundled here - it would be downloaded directly from them.
    echo.
    set /p "DL=  Download it now? [Y/n] "
    if /i not "!DL!"=="n" (
        echo  [..] Downloading Rubber Band...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { $u='https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip'; $z=Join-Path $env:TEMP 'rb.zip'; $d=Join-Path $env:TEMP 'rbx'; Invoke-WebRequest -Uri $u -OutFile $z -UseBasicParsing; if (Test-Path $d) { Remove-Item $d -Recurse -Force }; Expand-Archive $z $d; $e = Get-ChildItem $d -Recurse -Filter rubberband.exe | Select-Object -First 1; if (-not $e) { throw 'rubberband.exe not found inside the zip' }; New-Item -ItemType Directory -Force -Path 'bin' | Out-Null; Copy-Item $e.FullName 'bin\rubberband.exe' -Force; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
        if exist "bin\rubberband.exe" (
            echo  [OK] Rubber Band installed into bin\
        ) else (
            echo  [X] Automatic download did not work. Do it by hand:
            echo       1. open https://breakfastquay.com/rubberband/
            echo       2. download the command-line utility zip
            echo       3. copy rubberband.exe from it into the bin\ folder here
        )
    )
)

echo.
echo  ============================================
"%PY%" -m speedman.cli doctor
echo.
echo  Setup finished. To test:
echo    drag an audio file onto speedman.bat
echo  or open a terminal here and run:
echo    .venv\Scripts\activate
echo    speedman compare yourfile.mp3
echo.
pause
