@echo off
setlocal

rem Load test launcher for AI Anti-Fraud System.
rem Usage examples:
rem   stress_test.bat
rem   stress_test.bat core 500 50 10m
rem   stress_test.bat lifecycle 200 20 5m

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=core"

set "USERS=%~2"
if "%USERS%"=="" set "USERS=500"

set "SPAWN_RATE=%~3"
if "%SPAWN_RATE%"=="" set "SPAWN_RATE=50"

set "RUN_TIME=%~4"
if "%RUN_TIME%"=="" set "RUN_TIME=10m"

set "HOST=http://localhost:8000"
set "REPORT_DIR=reports"
set "STAMP=%DATE:~-4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "STAMP=%STAMP: =0%"
set "REPORT_NAME=%REPORT_DIR%\locust_%PROFILE%_%USERS%u_%STAMP%"

if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

echo.
echo Starting %PROFILE% load test
echo Host:       %HOST%
echo Users:      %USERS%
echo Spawn rate: %SPAWN_RATE% users/sec
echo Run time:   %RUN_TIME%
echo Report:     %REPORT_NAME%.html
echo.

set "LOCUST_PROFILE=%PROFILE%"
locust -f locustfile.py --host "%HOST%" --headless -u %USERS% -r %SPAWN_RATE% -t %RUN_TIME% --html "%REPORT_NAME%.html" --csv "%REPORT_NAME%"

endlocal
