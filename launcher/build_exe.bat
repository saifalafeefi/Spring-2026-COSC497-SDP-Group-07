@echo off
rem Build "Pulse Control.exe" into the repo root. Double-click this file.
rem
rem The exe is only the launcher window: every button runs the real code with the
rem project's virtualenv, so pulling new code never needs a rebuild. Rebuild only
rem when launcher\pulse_control.py itself changes.
rem
rem PyInstaller goes into a separate build environment, not the project's venv.

setlocal
set "HERE=%~dp0"
set "BASE=%USERPROFILE%\.venvs\sdp07\Scripts\python.exe"
set "BUILD=%LOCALAPPDATA%\pulse_control_build"

if not exist "%BASE%" (
  echo Can't find the project's Python at %BASE%
  echo Set up the virtualenv first: COMMANDS.md, section 1.
  pause
  exit /b 1
)

if not exist "%BUILD%\venv\Scripts\python.exe" (
  echo Creating the build environment...
  "%BASE%" -m venv "%BUILD%\venv" || goto :fail
)
echo Installing PyInstaller into the build environment...
"%BUILD%\venv\Scripts\python.exe" -m pip install --quiet --upgrade pip pyinstaller || goto :fail

echo Building...
"%BUILD%\venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "Pulse Control" ^
  --distpath "%HERE%.." ^
  --workpath "%BUILD%\work" ^
  --specpath "%BUILD%" ^
  "%HERE%pulse_control.py" || goto :fail

echo.
echo Done: "Pulse Control.exe" is in the project folder. Double-click it.
pause
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
