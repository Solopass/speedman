@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if "%~1"=="" (
    echo No file given - running the setup check instead.
    echo.
    echo To speed up a file, drag it onto this .bat file.
    echo.
    "%PY%" -m speedman.cli doctor
    echo.
    pause
    exit /b
)

echo Building the comparison set for "%~nx1"
echo.
"%PY%" -m speedman.cli compare "%~1" --speeds 5,6
if errorlevel 1 (
    echo.
    echo Something went wrong. Run setup.bat if you have not yet.
)
echo.
pause
