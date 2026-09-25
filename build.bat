@echo off
REM Builds a standalone ImagePacker.exe (no Python needed to run it).
pyinstaller --onefile --windowed --noconfirm --name ImagePacker --icon icon.ico --add-data "icon.ico;." app.py
echo.
echo Done. Executable is in dist\ImagePacker.exe
pause
