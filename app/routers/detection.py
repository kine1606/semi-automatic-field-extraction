from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.classifier_service import classify_image
from app.services.intake_service import create_detection_session

router = APIRouter(prefix="/detection", tags=["Detection"])


@router.post("/sessions")
async def create_session(
    file1: UploadFile | None = File(None),
    file2: UploadFile | None = File(None),
    file3: UploadFile | None = File(None),
):
    files = [f for f in [file1, file2, file3] if f is not None]

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    return await create_detection_session(files)


@router.post("/sessions/{session_id}/classify")
async def classify_session(session_id: str):
    session_dir = Path("storage") / session_id / "processed"

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    results = []

    for image_path in sorted(session_dir.glob("*.jpg")):
        result = classify_image(str(image_path))
        results.append({
            "imageId": image_path.stem,
            **result,
        })

    return {
        "sessionId": session_id,
        "status": "CLASSIFIED",
        "images": results,
    }