@echo off
setlocal enabledelayedexpansion
REM =====================================================================
REM Praxis Studio Launcher for Windows
REM Automatically locates Python in user directories, finds a collision-free
REM port (avoiding 8000 used by Artsylometrie/Nizaba), and launches the
REM interactive 3D Web Studio in your default browser.
REM =====================================================================

title Praxis Studio - 3D Manipulation and Physics Verifier
cd /d "%~dp0"

echo [Praxis Studio] Locating Python environment in Users directory...

set "PYTHON_EXE="

REM 1. Check user LocalAppData Python directory (standard install in Users)
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
)

REM 2. Check any Python 3.x in user LocalAppData
if not defined PYTHON_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" set "PYTHON_EXE=%%D\python.exe"
    )
)

REM 3. Check Python Launcher
if not defined PYTHON_EXE (
    py -3 --version >nul 2>&1
    if !errorlevel! == 0 set "PYTHON_EXE=py -3"
)

REM 4. Fallback to PATH python
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !errorlevel! == 0 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [Praxis Studio] ERROR: Python 3 was not found in your Users directory or PATH.
    echo Please verify Python is installed in %LOCALAPPDATA%\Programs\Python\
    pause
    exit /b 1
)

echo [Praxis Studio] Using Python: !PYTHON_EXE!

set PYTHONUNBUFFERED=1

REM Launch Praxis Studio runner with automatic free-port discovery & browser opening
if "!PYTHON_EXE!"=="py -3" (
    py -3 -u src\studio\run.py
) else if "!PYTHON_EXE!"=="python" (
    python -u src\studio\run.py
) else (
    "!PYTHON_EXE!" -u src\studio\run.py
)

if !errorlevel! neq 0 (
    echo [Praxis Studio] Server exited with code !errorlevel!.
    pause
)
