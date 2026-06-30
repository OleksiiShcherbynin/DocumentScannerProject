@echo off
:: Change to the directory where this .bat file is located
cd /d "%~dp0"
echo Starting RAG Document Assistant...
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
pause
