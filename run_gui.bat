@echo off
rem ===========================================================================
rem  run_gui.bat -- launch the lyric maker GUI
rem
rem  You can also drag an audio/video file onto this file to transcribe it
rem  directly and write a .lrc next to the media file.
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

if not "%~1"=="" (
  rem Drag-and-drop mode: transcribe the dropped file directly
  echo Using Python: !PYEXE!
  "!PYEXE!" "%~dp0lyric_maker.py" "%~1"
  if errorlevel 1 (
    echo.
    echo [FAILED] See the error message above. The .lrc was NOT written.
  ) else (
    echo.
    echo Done. The .lrc file sits next to the media file.
  )
  echo Tip: for songs, add --lyrics-file "lyrics.txt" for far better accuracy.
  pause
  exit /b 0
)

"!PYEXE!" "%~dp0lyric_gui.py"
if errorlevel 1 (
  echo.
  echo [FAILED] The GUI exited with an error. Try the command line form:
  echo   "!PYEXE!" lyric_maker.py "D:\path\song.mp3" --model small
  echo   "!PYEXE!" lyric_maker.py "D:\path\song.mp3" --lyrics-file "lyrics.txt"
  pause
)
exit /b 0

:detect
set "CAND=%~1"
%CAND% -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%i in ('%CAND% -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
exit /b 0
