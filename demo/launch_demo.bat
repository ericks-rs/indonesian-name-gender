@echo off
REM ============================================================
REM  Launch Demo Web App - Riset Nama Gender
REM  FastAPI + 8 neural models on RTX 5080
REM ============================================================

title Riset Gender Demo - FastAPI

cd /d "%~dp0"

echo.
echo ============================================================
echo   Riset Nama Gender - Web Demo
echo   Folder  : %CD%
echo   Backend : FastAPI + uvicorn
echo   Browser : http://127.0.0.1:8000
echo   API doc : http://127.0.0.1:8000/docs
echo ============================================================
echo.

REM Activate conda env (double-activation pattern)
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo [ERROR] Gagal init conda base.
    pause
    exit /b 1
)

call conda activate riset-gender
if errorlevel 1 (
    echo [ERROR] Gagal activate env "riset-gender".
    pause
    exit /b 1
)

echo [OK] Env activated: %CONDA_PREFIX%
echo.
echo Starting server... (Ctrl+C to stop)
echo Buka browser: http://127.0.0.1:8000
echo.

python -m uvicorn app:app --host 127.0.0.1 --port 8000

pause
