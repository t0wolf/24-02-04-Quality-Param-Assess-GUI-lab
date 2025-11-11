@echo off
setlocal enabledelayedexpansion
:: 设置程序路径和日志文件路径
set "PROGRAM_PATH=D:\Code\24-02-04-Quality-Param-Assess-GUI\24-02-04-Quality-Param-Assess-GUI\release\24-02-04-Quality-Param-Assess-GUI.exe"
set "LOG_PATH=C:\Users\86278\Desktop\monitor_log.txt"
set "PROGRAM_NAME=24-02-04"
set "PROGRAM_FULL_NAME=24-02-04-Quality-Param-Assess-GUI.exe"
set "CHECK_INTERVAL=25"  :: 检查间隔时间（秒）

:: 检查程序路径是否存在
if not exist "%PROGRAM_PATH%" (
    echo ERROR: Program path "%PROGRAM_PATH%" does not exist. Please check the path. >> "%LOG_PATH%"
    exit /b 1
)

:: 初始化日志文件（使用 UTF-8 编码）
if not exist "%LOG_PATH%" (
    echo. > "%LOG_PATH%"
    echo Log file created on %date% %time% > "%LOG_PATH%"
)

:loop
:: 获取当前时间
for /f "tokens=1-4 delims=:. " %%a in ("%time%") do (
    set "hour=%%a"
    set "minute=%%b"
)

:: 记录当前时间
echo [%date% %time%] Current hour: %hour%, minute: %minute% >> "%LOG_PATH%"

:: 检查当前时间是否在运行时间段内（8:00-12:00 或 14:00-18:00）
if %hour% geq 8 (
    if %hour% lss 12 (
        set "run=true"
    ) else if %hour% geq 14 (
        if %hour% lss 18 (
            set "run=true"
        ) else (
            set "run=false"
        )
    ) else (
        set "run=false"
    )
) else (
    set "run=false"
)

:: 记录运行状态
echo [%date% %time%] Run status: %run% >> "%LOG_PATH%"

:: 如果在运行时间段内
if "%run%"=="true" (
    :: 检查程序是否在运行
    set "output="
    for /f "delims=" %%i in ('tasklist ^| findstr /i "!PROGRAM_NAME!"') do (
        set "output=%%i"
    )
    echo output:!output!.
    if not defined output (
        :: 程序未运行，启动程序
        echo [%date% %time%] Program is not running, starting... >> "%LOG_PATH%"
        start "" "%PROGRAM_PATH%"
        :: 等待程序启动
        timeout /t 5 /nobreak >nul
    )
) else (
    :: echo Program should not be running, ERRORLEVEL %ERRORLEVEL% >> "%LOG_PATH%"
    :: 不在运行时间段内，关闭程序
    set "output="
    for /f "delims=" %%i in ('tasklist ^| findstr /i "!PROGRAM_NAME!"') do (
        set "output=%%i"
    )
    if defined output (
        echo [%date% %time%] Program is running, closing... >> "%LOG_PATH%"
        taskkill /f /im "%PROGRAM_FULL_NAME%" >nul 2>&1
        :: 等待程序关闭
        timeout /t 5 /nobreak >nul
    )
)

:: 等待一段时间后重新检查
timeout /t %CHECK_INTERVAL% /nobreak >nul
goto loop