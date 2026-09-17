# F1 Constructor Predictor

## Quick Start

### Backend (FastAPI)
```bash
cd "/Users/prachikarkhanis/Python Projects/F1 project"
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
```

### Frontend (React)
```bash
cd "/Users/prachikarkhanis/Python Projects/F1 project/frontend"
npm install
npm run dev
```

## Local Addresses
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Health check: `http://localhost:8000/health`

## Notes
- Run the backend and frontend in separate terminals.
- If port `8000` is already in use, stop the existing server or run:
  ```bash
  uvicorn backend.app:app --reload --port 8001
  ```
  Then update `API_BASE` in `frontend/src/App.jsx` to `http://localhost:8001`.
