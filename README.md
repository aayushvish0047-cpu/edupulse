# EduPulse

EduPulse is a full-stack academic trajectory and career skill-gap demo. The React
frontend collects a student profile and displays predictions from a FastAPI
backend. The backend uses a scikit-learn model trained on synthetic data.

## Requirements

- Python 3.10 or later
- Node.js 20.19+ or 22.12+ (required by Vite 8)
- npm

## Run locally on Windows

Open two PowerShell terminals in the `edupulse` directory.

### 1. Set up and start the backend

In the first terminal:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Training creates `backend/model.joblib` and `backend/students_dataset.csv`.
The API requires the model file, so run `python train.py` before starting
Uvicorn. API health and interactive documentation are available at
<http://localhost:8000/api/health> and <http://localhost:8000/docs>.

### 2. Set up and start the frontend

In the second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The default frontend API address is
`http://localhost:8000/api`. To override it, copy `frontend/.env.example` to
`frontend/.env.local`, edit `VITE_API_URL`, then restart the Vite server. If the
frontend origin changes, add it to `ALLOWED_ORIGINS` in `backend/main.py`.

## Features

- CGPA prediction and academic risk tier
- Target CGPA gap estimate and prediction explanations
- Career subject benchmarks, skill gaps, and recovery plan
- What-if simulator for attendance and weekly study hours

The API exposes `GET /api/health`, `GET /api/careers`, `POST /api/analyze`, and
`POST /api/simulate`.

## Notes

- Predictions are illustrative only. The model is trained on synthetic data
  and is not validated for real student outcomes.
- The generated model and dataset are local artifacts; retrain with
  `python train.py` whenever you want a fresh model.
