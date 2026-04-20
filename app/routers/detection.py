from fastapi import APIRouter, File, HTTPException, UploadFile

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