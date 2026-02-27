@echo off
setlocal enabledelayedexpansion

:: 1. Define the Project Path
set PROJECT_DIR=C:\AIlearning\local-rag
cd /d "%PROJECT_DIR%"

echo [INFO] Looking for Python engine...

:: 2. CHECK 1: The Local Virtual Environment (The best option)
if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    set TARGET_PYTHON="%PROJECT_DIR%\venv\Scripts\python.exe"
    echo [FOUND] Local Virtual Environment: !TARGET_PYTHON!
    goto :RUN_APP
)

:: 3. CHECK 2: Is 'py' (Python Launcher) available?
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set TARGET_PYTHON=py
    echo [FOUND] Windows Python Launcher (py)
    goto :RUN_APP
)

:: 4. CHECK 3: Is 'python' in the system PATH?
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set TARGET_PYTHON=python
    echo [FOUND] System Python (python)
    goto :RUN_APP
)

:: 5. ERROR: If nothing is found
echo [ERROR] Could not find any Python engine. 
echo Please ensure Python is installed or your venv exists in: %PROJECT_DIR%\venv
pause
exit

:RUN_APP
echo [LAUNCH] Starting Streamlit...
%TARGET_PYTHON% -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [CRASH] Streamlit failed to start. This usually means 'streamlit' is not 
    echo         installed in the Python engine found above.
    pause
)