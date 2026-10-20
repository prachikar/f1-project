# F1 Constructor Champion Predictor

Predicts the Formula 1 constructors' champion with a Random Forest trained on season standings (1959-2025). A FastAPI backend serves the model and a React (Vite) frontend lets you run predictions, race-by-race breakdowns and multi-year forecasts.

## Live deployment

| Part | Host | URL |
|---|---|---|
| Frontend (React/Vite) | Vercel | https://f1-project-sable.vercel.app |
| Backend (FastAPI) | Render (free plan) | https://f1-project-backend-jzob.onrender.com |
| Health check | Render | https://f1-project-backend-jzob.onrender.com/health |

Both auto-deploy on every push to `main`.

The Render free plan sleeps after about 15 minutes of inactivity, so the first request after idle can take 30-60 seconds. That is expected, not an error.

## Project layout

```
backend/app.py     FastAPI API (train, predict, forecast, race breakdown, standings)
frontend/          React + Vite UI
src/               Data loading, feature engineering, training, evaluation
data/              Historical F1 CSVs
render.yaml        Render blueprint for the backend
app.py, main.py    Original Streamlit app and CLI evaluation script
tests/             Regression tests for the API
```

## Model presets

Each preset in `frontend/src/App.jsx` sets a `train_end_year`. The model trains on seasons up to that year and is evaluated on every season after it.

| Preset | Trees / depth | Trains on | Evaluated on |
|---|---|---|---|
| Fast | 150 / 4 | 1959-2000 | 2001-2025 |
| Balanced | 300 / 6 | 1959-2010 | 2011-2025 |
| Deep | 500 / 10 | 1959-2020 | 2021-2025 |

1959 is the first season with a previous-season record to build lag features from. Deep's test window is only 5 seasons, so its accuracy is noisier than the other two.

## Run locally

Backend (from the repo root):
```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
```

Frontend (in a second terminal):
```bash
cd frontend
npm install
npm run dev
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000` (health check at `/health`)

Optional extras:
```bash
pip install -r requirements.txt && streamlit run app.py   # Streamlit version
python main.py                                            # print accuracy in the terminal
pytest tests/                                             # API regression tests
```

If port 8000 is busy, run the backend on another port, for example `uvicorn backend.app:app --reload --port 8001`, and start the frontend with `VITE_API_BASE=http://localhost:8001 npm run dev`.

## Configuration

| Variable | Where | Purpose |
|---|---|---|
| `VITE_API_BASE` | Vercel (build time) | Backend URL the frontend calls. Defaults to `http://localhost:8000`. Set on Vercel to the Render URL. |
| `FRONTEND_ORIGIN` | Render | Extra allowed CORS origin (comma-separated). `localhost:5173` and any `*.vercel.app` origin are already allowed. |

`VITE_API_BASE` is read at build time, so changing it on Vercel needs a redeploy.

## Deploying

- **Frontend:** import the repo in Vercel, set Root Directory to `frontend` and add `VITE_API_BASE`.
- **Backend:** Render reads `render.yaml` (build `pip install -r backend/requirements.txt`, start `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`).
