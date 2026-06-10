@echo off
cd /d D:\Distributed_RAG_pipeline
echo Starting backend...
start "RAG-Backend" cmd /c "python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1"
timeout /t 7 /nobreak >nul

cd /d D:\Distributed_RAG_pipeline\frontend
echo Starting frontend...
start "RAG-Frontend" cmd /c "npx vite --host --port 5173"

echo.
echo === Servers starting ===
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
timeout /t 5 /nobreak >nul
exit