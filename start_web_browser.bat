@echo off
REM =====================================================================
REM Praxis Studio Launcher for Windows
REM Starts the FastAPI backend with Uvicorn and opens the interactive
REM 3D physical manipulation viewer in your default web browser.
REM =====================================================================

title Praxis Studio — 3D Manipulation & Physics Verifier
cd /d "%~dp0"

echo [Praxis Studio] Checking Python environment...

REM Find Python executable
set PYTHON_CMD=python
py -3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py -3
)

%PYTHON_CMD% -c "import fastapi, uvicorn, mujoco, trimesh" >nul 2>&1
if %errorlevel% neq 0 (
    echo [Praxis Studio] Installing required web and physics dependencies...
    %PYTHON_CMD% -m pip install fastapi uvicorn[standard] mujoco trimesh
)

echo [Praxis Studio] Starting server on http://127.0.0.1:8000 ...
echo [Praxis Studio] Press Ctrl+C in this terminal to stop the server.

REM Open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

REM Launch FastAPI Studio Server
%PYTHON_CMD% -m uvicorn src.studio.server:app --host 127.0.0.1 --port 8000

pause
