@echo off
rem ===========================================================================
rem  install_deps.bat -- install Python dependencies for the lyric maker
rem
rem  All packages come from the Tsinghua PyPI mirror, because pypi.org and
rem  github.com are unreachable on this network.
rem
rem  IMPORTANT: the interpreter chosen here is recorded in .python-path.txt and
rem  reused by the other .bat files. The compiled wheels (ctranslate2, av,
rem  numpy, onnxruntime) are ABI-locked to one Python minor version, so the
rem  launchers MUST use the same interpreter that installed them.
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "HF_ENDPOINT=https://hf-mirror.com"
if not exist ".tmp" mkdir ".tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"

set "PYEXE="
call :detect "py -3.12"
if not defined PYEXE call :detect "py -3.11"
if not defined PYEXE call :detect "py -3.10"
if not defined PYEXE call :detect "py -3"
if not defined PYEXE call :detect "python"
if not defined PYEXE call :detect "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

if not defined PYEXE (
  echo.
  echo [ERROR] No usable Python 3.9+ found.
  echo         Install Python 3.9+ ^(3.12 recommended^) from python.org and
  echo         tick "Add python.exe to PATH", then run this script again.
  echo.
  pause
  exit /b 1
)

echo Using Python : !PYEXE!
!PYEXE! -c "import sys;print('Version      :', sys.version.split()[0])"
echo Installing to: %~dp0.deps
echo.

> ".python-path.txt" echo !PYEXE!

"!PYEXE!" -m pip install --target "%~dp0.deps" --upgrade ^
  --disable-pip-version-check --no-warn-script-location ^
  faster-whisper mutagen opencc-python-reimplemented ^
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60

if errorlevel 1 (
  echo.
  echo [FAILED] Dependency installation failed. See the messages above.
  pause
  exit /b 1
)

echo.
echo [OK] Dependencies installed.
echo      Next step: run download_model.bat
pause
exit /b 0

:detect
rem %~1 = candidate command line. Accepts it only if it is Python 3.9+.
set "CAND=%~1"
%CAND% -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%i in ('%CAND% -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
exit /b 0
