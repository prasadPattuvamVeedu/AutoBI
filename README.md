# AutoBI AI

Initial project setup for AutoBI AI.

This repository currently contains only the starter structure, settings, routing, and placeholder files. Full authentication, dataset upload, profiling, previews, dashboards, ML, charts, exports, AWS, Spark, and big-data features are intentionally not implemented yet.

## Backend Setup

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

## Local URLs

- Backend API: http://localhost:8000/api
- Frontend: http://localhost:5173

## Current Scope

- React + Vite frontend skeleton
- Django + Django REST Framework backend skeleton
- JWT login and refresh URL wiring through SimpleJWT
- SQLite configuration
- Local media configuration
- Placeholder accounts and datasets apps
