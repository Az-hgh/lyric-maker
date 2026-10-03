@echo off
rem ===========================================================================
rem  download_model.bat -- fetch a faster-whisper model
rem
rem  Usage:  download_model.bat             list models and status
rem          download_model.bat small       download (hf-mirror)
rem          download_model.bat large-v3    download (ModelScope, recommended)
rem
rem  Uses the interpreter recorded by install_deps.bat so that the compiled
rem  dependencies stay ABI-compatible.
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "HF_ENDPOINT=https://hf-mirror.com"
if not exist ".tmp" mkdir ".tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PYTHONPATH=%~dp0.deps"

set "PYEXE="
if exist ".python-path.txt" (
  set /p PYEXE=<".python-path.txt"
  if defined PYEXE if not exist "!PYEXE!" set "PYEXE="
)
if not defined PYEXE call :detect "py -3.12"
if not defined PYEXE call :detect "py -3.11"
if not defined PYEXE call :detect "py -3.10"
if not defined PYEXE call :detect "py -3"
if not defined PYEXE call :detect "python"
if not defined PYEXE call :detect "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

if not defined PYEXE (
  echo [ERROR] No usable Python 3.9+ found. Run install_deps.bat first.
  pause
  exit /b 1
)
if not exist "%~dp0.deps\faster_whisper" (
  echo [ERROR] Dependencies missing. Run install_deps.bat first.
  pause
  exit /b 1
)

"!PYEXE!" "%~dp0get_model.py" %*

if errorlevel 1 (
  echo.
  echo [FAILED] Model download did not complete. Progress is kept in a .part
  echo          file, so simply run this script again to resume.
  pause
  exit /b 1
)
pause
exit /b 0

:detect
set "CAND=%~1"
%CAND% -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%i in ('%CAND% -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
exit /b 0
