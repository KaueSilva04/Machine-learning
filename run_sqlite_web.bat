@echo off
REM Executa sqlite-web para o banco qrcode_data.db
cd /d "%~dp0"
python -m sqlite_web --host 127.0.0.1 --port 8080 qrcode_data.db
