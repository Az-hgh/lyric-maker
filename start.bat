@echo off
rem ===========================================================================
rem  start.bat -- lyric-maker 的唯一入口 / the only entry point
rem
rem  Everything is inside this one launcher: it locates Python, installs the
rem  missing dependencies, downloads a speech model if there is none, and then
rem  starts the GUI or processes the file you dropped on it.
rem  There are no other launcher scripts -- all the work is in lyric_maker.py,
rem  and this file only prepares the environment and passes arguments through.
rem
rem  Usage:
rem    start.bat                            open the GUI
rem    start.bat "video.mp4"                process a file directly
rem    start.bat "video.mp4" --subs track   only use existing subtitles
rem    start.bat "video.mp4" --subs ocr     read on-screen hard subtitles
rem    start.bat "video.mp4" --format srt   output .srt instead of .lrc
rem    start.bat --install-ocr              install the optional OCR deps
rem    start.bat --get-model large-v3       download a specific model
rem    start.bat --setup                    prepare everything, then exit
rem    start.bat --help                     show the engine help
rem
rem  NOTE: keep this file ASCII-only. cmd.exe reads .bat with the OEM code page
rem  (936 here), so UTF-8 Chinese in comments gets mis-decoded and can even be
rem  executed as a command.
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "HF_ENDPOINT=https://hf-mirror.com"
if not exist ".tmp" mkdir ".tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PYTHONPATH=%~dp0.deps"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

if /i "%~1"=="--help"        goto :help
if /i "%~1"=="-h"            goto :help
if /i "%~1"=="/?"            goto :help
if /i "%~1"=="--install-ocr" goto :install_ocr

call :prepare
if errorlevel 1 exit /b 1

if /i "%~1"=="--setup" (
  echo.
  echo [OK] Everything is ready. Double-click start.bat next time to use the GUI.
  pause
  exit /b 0
)

echo.
if "%~1"=="" (
  echo Launching the GUI...
  echo.
  "!PYEXE!" "%~dp0lyric_maker.py" --gui
  if errorlevel 1 (
    echo.
    echo [FAILED] The GUI exited with an error. Command line form:
    echo     start.bat "D:\path\song.mp3"
    echo     start.bat "D:\path\song.mp3" --subs ocr
    pause
  )
) else (
  echo Processing: %~1
  echo.
  "!PYEXE!" "%~dp0lyric_maker.py" %*
  if errorlevel 1 (
    echo.
    echo [FAILED] See the message above. Nothing was written.
  ) else (
    echo.
    echo Done. The output file sits next to the media file.
  )
  pause
)
exit /b 0


rem ===========================================================================
rem  Preparation: Python -> main deps -> model -> (optional OCR notice)
rem ===========================================================================
:prepare
call :prepare_python
if errorlevel 1 exit /b 1

rem ---- main dependencies ----
if not exist "%~dp0.deps\faster_whisper" (
  echo.
  echo [SETUP 1/3] Main dependencies missing. Installing, about 150 MB...
  echo             source: Tsinghua PyPI mirror
  echo.
  "!PYEXE!" -m pip install --target "%~dp0.deps" --upgrade --disable-pip-version-check --no-warn-script-location faster-whisper mutagen opencc-python-reimplemented -i %PIP_MIRROR% --timeout 60
  if not exist "%~dp0.deps\faster_whisper" (
    echo.
    echo [ERROR] Dependency installation did not finish.
    pause
    exit /b 1
  )
  rem record the interpreter: the compiled wheels are ABI-locked to it
  > ".python-path.txt" echo !PYEXE!

rem ---------------------------------------------------------------
rem  yt-dlp for the in-app video downloader. Installed into its own
rem  directory: it pulls in websockets / pycryptodomex, and mixing
rem  those into .deps tends to break the verified versions.
rem  Failure here is NOT fatal - the download page then just tells
rem  you yt-dlp is missing.
rem ---------------------------------------------------------------
if not exist "%~dp0.deps-ytdlp\yt_dlp" (
  echo.
  echo [SETUP 2/3] Installing yt-dlp for the video downloader, about 20 MB...
  echo.
  "!PYEXE!" -m pip install --target "%~dp0.deps-ytdlp" --upgrade --disable-pip-version-check --no-warn-script-location yt-dlp -i %PIP_MIRROR% --timeout 60
  if not exist "%~dp0.deps-ytdlp\yt_dlp" (
    echo.
    echo [WARN] yt-dlp download failed; the video download page will say it is missing.
  )
)

rem ---------------------------------------------------------------
rem  yt-dlp-ejs: JS solver for YouTube's "n challenge". Without it
rem  YouTube only exposes a few low-res formats. Also non-fatal.
rem ---------------------------------------------------------------
if not exist "%~dp0.deps-ejs\yt_dlp_ejs" (
  echo.
  echo [SETUP 3/3] Installing yt-dlp-ejs (YouTube challenge solver)...
  echo.
  "!PYEXE!" -m pip install --target "%~dp0.deps-ejs" --upgrade --disable-pip-version-check --no-warn-script-location yt-dlp-ejs -i %PIP_MIRROR% --timeout 60
  if not exist "%~dp0.deps-ejs\yt_dlp_ejs" (
    echo.
    echo [WARN] yt-dlp-ejs download failed; YouTube may only offer low-res formats.
  )
)
)

rem ---- speech model ----
set "HAVE_MODEL=0"
for /d %%d in ("%~dp0models\*") do (
  if exist "%%~fd\model.bin" set "HAVE_MODEL=1"
)
if "!HAVE_MODEL!"=="0" (
  echo.
  echo [SETUP 2/2] No speech model yet. Downloading "small", about 480 MB...
  echo             For better quality later:  start.bat --get-model large-v3
  echo.
  "!PYEXE!" "%~dp0lyric_maker.py" --get-model small
  set "HAVE_MODEL=0"
  for /d %%d in ("%~dp0models\*") do (
    if exist "%%~fd\model.bin" set "HAVE_MODEL=1"
  )
  if "!HAVE_MODEL!"=="0" (
    echo.
    echo [ERROR] Model download did not finish. Run this launcher again to resume.
    pause
    exit /b 1
  )
)

rem ---- optional OCR ----
if not exist "%~dp0.deps-ocr\rapidocr_onnxruntime" (
  echo.
  echo [NOTE] On-screen hard-subtitle OCR is not installed.
  echo        Subtitle tracks and sidecar .srt/.ass files work without it.
  echo        To also read subtitles burned into the picture, run:
  echo            start.bat --install-ocr
)
exit /b 0


rem ===========================================================================
rem  Python 3.9+ detection (sets PYEXE)
rem ===========================================================================
:prepare_python
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
  echo.
  echo [ERROR] No usable Python 3.9+ found.
  echo         Install Python 3.12 from python.org, tick "Add python.exe to PATH",
  echo         then run this launcher again.
  pause
  exit /b 1
)
exit /b 0


rem ===========================================================================
rem  --install-ocr : optional, only needed for on-screen hard subtitles
rem ===========================================================================
:install_ocr
call :prepare_python
if errorlevel 1 exit /b 1
if not exist "%~dp0.deps\faster_whisper" (
  echo.
  echo [ERROR] Main dependencies are missing. Run start.bat first.
  pause
  exit /b 1
)
echo.
echo Installing OCR dependencies into .deps-ocr, about 17 MB...
echo source: Tsinghua PyPI mirror
echo.
"!PYEXE!" -m pip install --target "%~dp0.deps-ocr" --no-deps --upgrade --disable-pip-version-check --no-warn-script-location rapidocr-onnxruntime shapely pyclipper -i %PIP_MIRROR% --timeout 60
if not exist "%~dp0.deps-ocr\rapidocr_onnxruntime" (
  echo.
  echo [FAILED] OCR dependency installation failed. See the messages above.
  pause
  exit /b 1
)
echo.
echo [OK] OCR dependencies installed. Hard-subtitle OCR is now available.
pause
exit /b 0


rem ===========================================================================
rem  --help : pass through to the engine, which prints the full option list
rem ===========================================================================
:help
call :prepare_python
if errorlevel 1 exit /b 1
"!PYEXE!" "%~dp0lyric_maker.py" --help
echo.
pause
exit /b 0


rem ===========================================================================
rem  Pick the first candidate that is Python 3.9+ and store its path in PYEXE
rem ===========================================================================
:detect
set "CAND=%~1"
%CAND% -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%i in ('%CAND% -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
exit /b 0
