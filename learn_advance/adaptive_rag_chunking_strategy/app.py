"""Small FastAPI upload UI for the adaptive chunking strategy selector."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

try:  # Works both with `uvicorn app:app` from this folder and package imports.
    from adaptive_rag_chunking_strategy import (
        DocumentReadError,
        DOCUMENT_EXTENSIONS,
        read_document_bytes,
        select_chunking_strategy,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by package-style startup.
    from .adaptive_rag_chunking_strategy import (
        DocumentReadError,
        DOCUMENT_EXTENSIONS,
        read_document_bytes,
        select_chunking_strategy,
    )


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR.parent / "sample-data"
SAMPLE_FILES = {
    "structure_based": "adaptive-structure.md",
    "recursive": "adaptive-recursive.txt",
    "semantic": "adaptive-semantic.txt",
    "fixed_size": "adaptive-fixed-size.txt",
    "agentic": "adaptive-agentic.txt",
}
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="Adaptive RAG Chunking Strategy Selector")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "error": None})


def analyze_upload(file_name: str, payload: bytes) -> dict:
    content, metadata = read_document_bytes(payload, file_name)
    result = select_chunking_strategy(
        file_name=file_name,
        content=content,
        metadata=metadata,
    )
    # Do not return the extracted document text to the browser by default.
    return {
        "file_name": file_name,
        "strategy": result["strategy"],
        "reason": result["reason"],
        "confidence": result["confidence"],
        "signals": result["signals"],
    }


@app.post("/api/analyze")
async def analyze_api(file: UploadFile = File(...)):
    payload = await file.read()
    try:
        return analyze_upload(file.filename or "uploaded-file", payload)
    except DocumentReadError as error:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_form(request: Request, file: UploadFile = File(...)):
    result = None
    error = None
    try:
        result = analyze_upload(file.filename or "uploaded-file", await file.read())
    except DocumentReadError as exception:
        error = str(exception)
    return templates.TemplateResponse(request, "index.html", {"result": result, "error": error})


@app.post("/analyze-sample", response_class=HTMLResponse)
async def analyze_sample(request: Request, strategy: str = Form(...)):
    result = None
    error = None
    sample_name = SAMPLE_FILES.get(strategy)
    if not sample_name:
        error = "Unknown sample strategy."
    else:
        sample_path = SAMPLE_DIR / sample_name
        try:
            result = analyze_upload(sample_name, sample_path.read_bytes())
        except (DocumentReadError, OSError) as exception:
            error = str(exception)
    return templates.TemplateResponse(request, "index.html", {"result": result, "error": error})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


SUPPORTED_EXTENSIONS = ", ".join(f".{extension}" for extension in sorted(DOCUMENT_EXTENSIONS))
