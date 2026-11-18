@echo off
REM PharmaML Studio — the one-command Windows installer build (part9 §16).
REM Thin cmd wrapper around the real pipeline (build_windows.ps1); every
REM failure propagates a nonzero exit code.
REM
REM   packaging\windows\build_windows.bat           (full release build)
REM   packaging\windows\build_windows.bat --dev     (skips the test gate;
REM                                                  NEVER for a release)

setlocal
set "ARGS="
if /i "%1"=="--dev" set "ARGS=-SkipTests"

powershell -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0build_windows.ps1" %ARGS%
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (
    echo BUILD OK
) else (
    echo BUILD FAILED with exit code %RC% >&2
)
endlocal & exit /b %RC%
