import sys
from pathlib import Path

from fastapi import FastAPI

# Allow running this file directly: `python app/main.py`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.routers.detection import router as detection_router

app = FastAPI(title="Equipment Vision Service")

app.include_router(detection_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
