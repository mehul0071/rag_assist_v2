from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.settings import settings
from app.routers import conversations, documents, evaluation, query
from app.core.observability.metrics import setup_metrics
from app.core.observability.tracing import setup_tracing


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG Assistant API is running 🚀"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

app.include_router(documents.router)
app.include_router(conversations.router)
app.include_router(query.router)
app.include_router(evaluation.router)

setup_metrics(app)
setup_tracing()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)