import io
import os
from typing import Annotated
import zipfile
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.core.container import get_container
from app.config.settings import settings

router = APIRouter(prefix="/api/documents", tags=["Documents"])


def is_safe_path(base_dir: str, target_path: str) -> bool:
    base_dir_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(target_path)
    return target_abs.startswith(base_dir_abs)


@router.post("/upload")
async def upload_documents(
    file: Annotated[UploadFile, File(...)],
    container=Depends(get_container)
):
    if not file.filename or not file.filename.lower().endswith('.zip'):
        raise HTTPException(
            status_code=400,
            detail="Only .zip files are allowed."
        )

    upload_dir = settings.KNOWLEDGE_BASE
    os.makedirs(upload_dir, exist_ok=True)

    try:
        content = await file.read()
        extracted_count = 0

        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            for member in zip_ref.namelist():
                if member.startswith('__MACOSX') or os.path.basename(member).startswith('.'):
                    continue

                if not member.lower().endswith(('.pdf', '.txt', '.md')):
                    continue

                target_path = os.path.join(upload_dir, member)
                
                if not is_safe_path(upload_dir, target_path):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid zip file: Path traversal detected."
                    )

                zip_ref.extract(member, upload_dir)
                extracted_count += 1

        if extracted_count == 0:
            return {"message": "No supported files (.pdf, .txt, .md) found.", "extracted": 0}

        result = await container.rag_service.ingest_folder(upload_dir)

        return {
            "message": f"Success! {extracted_count} document(s) uploaded and indexed.",
            "extracted_files": extracted_count,
            "ingestion_result": result
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/refresh")
async def refresh_knowledge_base(container=Depends(get_container)):
    result = await container.rag_service.ingest_folder()
    return {
        "status": "success",
        "message": "Knowledge base refreshed.",
        "ingestion_result": result
    }