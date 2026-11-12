@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo ==========================================================
echo   GROUP PORTRAIT - FULL AUTOMATIC SETUP
echo ==========================================================
echo.
echo This window will do everything by itself:
echo   1. find or install Python 3.15
echo   2. install libraries
echo   3. build GroupPortrait.exe
echo.
echo Internet is needed. Do not close this window.
echo It takes 15-30 minutes.
echo.
pause

set "PYEXE="
set "PYDIR="

call :findpython
if defined PYEXE goto havepython

echo.
echo [1/4] Python not found. Downloading Python 3.15.2 ...
set "SETUP=%TEMP%\python-3.15.2-amd64.exe"
if exist "%SETUP%" del /q "%SETUP%"

curl -L -o "%SETUP%" "https://www.python.org/ftp/python/3.15.2/python-3.15.2-amd64.exe"
if not exist "%SETUP%" goto dlfail
for %%A in ("%SETUP%") do set "SZ=%%~zA"
if !SZ! LSS 20000000 goto dlfail
echo     Downloaded. Installing quietly, please wait ...
"%SETUP%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
if errorlevel 1 echo     Installer returned an error, checking anyway ...

call :findpython
if not defined PYEXE goto nopython

:havepython
echo.
echo     Python found: !PYEXE!
"!PYEXE!" -c "import sys; print('    Version:', sys.version.split()[0])"

echo.
echo [2/4] Updating pip ...
"!PYEXE!" -m pip install --upgrade pip --quiet
"!PYEXE!" -m pip install --upgrade setuptools wheel --quiet

echo.
echo [3/4] Installing libraries (10-20 minutes) ...
"!PYEXE!" -m pip install -r requirements.txt
if errorlevel 1 goto libfail

echo.
echo     Checking libraries ...
"!PYEXE!" -c "import PySide6, cv2, mediapipe, numpy, PIL; print('    Libraries OK')"
if errorlevel 1 goto libfail

echo.
echo [4/4] Building program (5-10 minutes) ...
"!PYEXE!" -m PyInstaller --noconfirm --clean GroupPortrait.spec
if errorlevel 1 goto buildfail

echo.
echo ==========================================================
echo   DONE
echo ==========================================================
echo.
echo Your program:
echo   %~dp0dist\GroupPortrait\GroupPortrait.exe
echo.
echo You can copy the whole folder dist\GroupPortrait
echo to the desktop or to another computer.
echo.
start "" explorer "%~dp0dist\GroupPortrait"
pause
exit /b 0

:findpython
set "PYEXE="
for %%V in (315 314 313) do (
  if not defined PYEXE (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
  )
)
if not defined PYEXE (
  if exist "C:\Python315\python.exe" set "PYEXE=C:\Python315\python.exe"
)
if not defined PYEXE (
  if exist "C:\Program Files\Python315\python.exe" set "PYEXE=C:\Program Files\Python315\python.exe"
)
if not defined PYEXE (
  for %%P in (python.exe) do (
    for /f "delims=" %%F in ('where %%P 2^>nul') do (
      if not defined PYEXE set "PYEXE=%%F"
    )
  )
)
if defined PYEXE (
  "!PYEXE!" -c "import sys; assert sys.version_info[:2] >= (3,13)" >nul 2>&1
  if errorlevel 1 set "PYEXE="
)
exit /b 0

:dlfail
echo.
echo ==========================================================
echo   ERROR: could not download Python
echo ==========================================================
echo.
echo Check the internet connection and run setup.bat again.
echo If a proxy or antivirus blocks it, turn it off for a while.
echo.
pause
exit /b 1

:nopython
echo.
echo ==========================================================
echo   ERROR: Python installed but not found
echo ==========================================================
echo.
echo Restart the computer and run setup.bat again.
echo.
pause
exit /b 1

:libfail
echo.
echo ==========================================================
echo   ERROR: could not install libraries
echo ==========================================================
echo.
echo Copy the text above (yellow and red lines) and send it.
echo Most often it is the internet or an antivirus.
echo.
pause
exit /b 1

:buildfail
echo.
echo ==========================================================
echo   Libraries installed, but the exe was not built
echo ==========================================================
echo.
echo The program still works: double-click run.bat
echo Copy the text above and send it to me.
echo.
pause
exit /b 1
