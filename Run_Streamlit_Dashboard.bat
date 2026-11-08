@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  SUSTAINABLE MANUFACTURING ANALYTICS - STREAMLIT DASHBOARD
echo ============================================================
echo.
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo The dashboard did not start. Check the error above.
  echo Try: python -m pip install -r requirements.txt
  pause
)
endlocal
