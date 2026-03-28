# ⚡ Basker Energy — World-Class SaaS Platform v2.0
**AI-Powered Renewable Energy Intelligence**
*Built by BCG Smart Energy Solutions for Basker Energy Project*

---

## 🚀 Quick Start (Local — 1 command)

```bash
docker-compose up
```

| Service | URL |
|---------|-----|
| 📊 Dashboard | http://localhost:8501 |
| 🔌 API | http://localhost:8000 |
| 📚 API Docs | http://localhost:8000/docs |
| 📈 Grafana | http://localhost:3000 |
| 🗄️ InfluxDB | http://localhost:8086 |

### Demo Credentials
| Role | Email | Password |
|------|-------|----------|
| Utility Admin | admin@baskerenergy.ai | Basker@2026 |
| MSME Manager | msme@baskerenergy.ai | MSME@2026 |
| Home User | home@baskerenergy.ai | Home@2026 |
| Demo | demo@baskerenergy.ai | demo123 |

---

## 📁 Project Structure

```
basker_saas/
├── streamlit_app/          # Frontend dashboard (Streamlit)
│   ├── main.py            # 7-module dashboard (1,200+ lines)
│   ├── ml_core.py         # ML engine (physics + AI/ML)
│   ├── .streamlit/config.toml
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                # FastAPI REST API
│   ├── main.py            # 14 REST endpoints + WebSocket
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml      # Full stack orchestration
└── .github/workflows/      # CI/CD pipeline
    └── deploy.yml
```

---

## 🤖 AI/ML Models

| Model | Algorithm | Task | Performance |
|-------|-----------|------|-------------|
| M001 | Random Forest (n=500) | AC Power Prediction | R²=0.9999 |
| M002 | XGBoost (n=150, lr=0.05) | AC Power Prediction | R²=0.9998 |
| M003 | LSTM (64→32→Dense) | 24h Forecasting | R²=0.9110 |
| M004 | XGBoost Classifier | Fault Detection | AUC=0.87 |
| M005 | Isolation Forest | Anomaly Detection | FPR<4% |
| M006 | XGBoost Regressor | Soiling Estimation | MAE=0.18% |
| M007 | RL (PPO) | BESS Dispatch | +₹4.2L/MW/yr |

---

## 🔌 API Reference

**Base URL:** `https://api.baskerenergy.ai/v1`
**Auth:** `Authorization: Bearer <token>`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/data/ingest | Ingest SCADA reading |
| POST | /v1/predict/power | Predict AC power (RF+XGB) |
| POST | /v1/predict/fault | Fault probability score |
| POST | /v1/predict/soiling | Soiling loss % |
| POST | /v1/forecast/dayahead | 24–168h generation forecast |
| GET | /v1/forecast/weekahead | 7-day outlook |
| GET | /v1/dashboard/summary | Master KPI summary |
| GET | /v1/dashboard/kpis | Real-time KPIs |
| POST | /v1/recommend | AI action recommendations |
| GET | /v1/trading/signals | Sell/hold/charge signals |
| GET | /v1/analytics/roi/{id} | ROI & financial analytics |
| GET | /v1/analytics/benchmark/{id} | Fleet benchmarking |
| WS | /v1/ws/live/{plant_id} | Real-time SCADA stream |

---

## ☁️ Cloud Deployment

### Option A: Render + Streamlit Cloud (Free)
1. Push repo to GitHub
2. Render: `New Web Service → backend/ → Python → uvicorn main:app`
3. Streamlit Cloud: `New App → streamlit_app/main.py`

### Option B: Docker (Any cloud)
```bash
docker build -t basker-api ./backend
docker build -t basker-dashboard ./streamlit_app
```

### Option C: Render Auto-Deploy (via CI/CD)
Add secrets to GitHub:
- `RENDER_API_KEY`
- `RENDER_SERVICE_ID_API`
- `RENDER_SERVICE_ID_DASH`

Push to `main` → auto-deploys ✅

---

## 📊 3-Tier Platform

| Tier | Segment | Price | Key Value |
|------|---------|-------|-----------|
| T1 | Utility Scale (>1MW) | ₹1.5–5L/MW/yr | +₹29.8L/MW/yr uplift |
| T2 | MSMEs & Factories | ₹15K–50K/site/mo | ₹54L/yr savings (500kW) |
| T3 | Residential Premium | ₹2K–10K/home/mo | ₹50K/yr savings (10kW) |

---

*Built by BCG Smart Energy Solutions for Basker Energy / Adani Energy*
*© 2026 Basker Energy. Confidential.*
