@echo off
echo Stopping Prism AI...
taskkill /f /fi "WINDOWTITLE eq Prism Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Prism Frontend*" >nul 2>&1
echo Done.
