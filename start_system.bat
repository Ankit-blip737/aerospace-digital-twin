@echo off
color 0b
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║    AEROSPACE DIGITAL TWIN — SWARM COMMAND CENTER v2     ║
echo  ║         5-UAV Fleet · 13 Sensor Datasets · AI Engine    ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

echo [1/3] Starting Python AI Engine (Port 8000)...
echo       → 5-Class Fault Detection (GRU Model + Real Scaler)
echo       → Remaining Useful Life (RUL) Endpoint Active
start "AI Engine [Port 8000]" cmd /k "uvicorn ml_api:app --reload --port 8000"

echo.
timeout /t 3 /nobreak >nul

echo [2/3] Starting Node.js Command Center (Port 5000)...
echo       → Loading 13 sensor datasets (ATT, BARO, BAT, GPS, IMU, MAG, MOTB, PSCD, RATE, VIBE, XKF1, CTUN, Fusion)
echo       → Independent Swarm Telemetry for 5 UAVs
echo       → MongoDB Atlas Time-Series Logging Active
start "Node Server [Port 5000]" cmd /k "node server.js"

echo.
timeout /t 3 /nobreak >nul

echo [3/3] Starting React Dashboard (Port 5173)...
echo       → Real-time Charts · Attitude Indicator · GPS Map
echo       → Fault Injection Console · RUL Panel
start "React UI [Port 5173]" cmd /k "cd frontend && npm run dev"

echo.
timeout /t 5 /nobreak >nul
echo  ✓ All systems online! Opening dashboard...
start http://localhost:5173

echo.
echo  ┌──────────────────────────────────────────────┐
echo  │  Services:                                   │
echo  │    AI Engine    →  http://localhost:8000     │
echo  │    Node Server  →  http://localhost:5000     │
echo  │    Dashboard    →  http://localhost:5173     │
echo  └──────────────────────────────────────────────┘
echo.
echo  You can close this window. All services run independently.