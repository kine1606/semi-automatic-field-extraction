# Equipment Vision Service

FastAPI starter project.

## Create and activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the API (development)

```powershell
uvicorn app.main:app --reload
```

## Test

- Health endpoint: `GET http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
