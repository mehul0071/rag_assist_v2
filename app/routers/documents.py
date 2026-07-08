import io
import os
from typing import Annotated
import zipfile
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.core.container import get_container
from app.config.settings import settings

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload")
async def upload_documents(
    file: Annotated[UploadFile, File(...)],
    container = Depends(get_container)
):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(
            status_code=400 , 
            detail="only zip file is allowed , please upload zip file"
        )
    
    upload_dir = settings.KNOWLEDGE_BASE
    os.makedirs(upload_dir, exist_ok=True)

    try:
        content = await file.read()
        extracted_count = 0

        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            for i in zip_ref.namelist():
                if i.startswith('__MACOSX') or os.path.basename(i).startswith('.'):
                    continue
                if i.lower().endswith(('.pdf', '.txt', '.md')):
                    zip_ref.extract(i, upload_dir)
                    extracted_count += 1

        if extracted_count == 0:
            return {"message": "No supported files found.", "extracted": 0}
        
        result = await container.rag_service.ingest_folder(upload_dir)

        return {
            "message": f"Success! {extracted_count} document(s) uploaded and indexed.",
            "extracted_files": extracted_count,
            "ingestion_result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/refresh")
async def refresh_knowledge_base(container = Depends(get_container)):
    result = await container.rag_service.ingest_folder()
    print(f"==================result=================={result}")
    return {
        "status": "success", 
        "message": "Knowledge base refreshed.",
        "ingestion_result": result
    }