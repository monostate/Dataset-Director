"""
FastAPI service for Dataset Director - Kumo SDK and HuggingFace integration.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .hf_export import export_to_huggingface
from .kumo_client import KumoClient
from .kumo_graph import build_graph
from .rfm_predict import get_coverage_pql, get_specs_pql
from .security import (
    SecurityHeadersMiddleware,
    limiter,
    safe_error_response,
    sanitize_text,
    session_manager,
    verify_api_key,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_TOTAL_ROWS = 100
TEMP_DIR = Path("/tmp/vibe-data-director")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Vibe Data Director",
    description="Dataset Director API for Kumo SDK and HuggingFace integration",
    version="0.1.0"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key authentication."""
    return verify_api_key(credentials)

# Pydantic Models
class SessionInitRequest(BaseModel):
    classes: List[str] = Field(..., min_items=1, max_items=10)
    target_count_per_class: int = Field(..., ge=1, le=100)
    styles: Optional[List[str]] = Field(default=["none"])
    include_negations: Optional[bool] = Field(default=False)

    @validator('classes')
    def validate_classes(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate classes not allowed")
        return v

class SessionInitResponse(BaseModel):
    session_id: str
    specs: List[Dict[str, Any]]

class SeedRow(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    class_name: str = Field(..., alias="class")  # 'class' is reserved in Python
    style: Optional[str] = Field(default="none")
    negation: Optional[bool] = Field(default=False)

class SeedUploadJSONRequest(BaseModel):
    session_id: str
    rows: List[SeedRow]

class SeedUploadResponse(BaseModel):
    rows: List[Dict[str, Any]]
    total_rows: int

class CoverageItem(BaseModel):
    class_name: str = Field(..., alias="class")
    pred_count: int

class CoverageResponse(BaseModel):
    coverage: List[CoverageItem]

class SpecsResponse(BaseModel):
    spec_ids: List[str]

class HFExportRequest(BaseModel):
    session_id: str
    repo_id: str = Field(..., pattern="^[a-zA-Z0-9-_]+/[a-zA-Z0-9-_]+$")

class HFExportResponse(BaseModel):
    repo_url: str

# Helper functions
def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"

def generate_sample_id() -> str:
    """Generate a unique sample ID."""
    return f"sample_{uuid.uuid4().hex[:8]}"

def generate_spec_id(class_name: str, style: str, negation: bool) -> str:
    """Generate a deterministic spec ID from class, style, and negation."""
    return f"{class_name}|{style}|{int(negation)}"

def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data or raise 404."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")
    return session

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vibe-data-director"}

@app.post("/session/init", response_model=SessionInitResponse)
@limiter.limit("10/minute")
async def init_session(
    request: Request,
    body: SessionInitRequest,
    _: str = Depends(verify_token)
) -> SessionInitResponse:
    """Initialize a new session with specs grid."""
    session_id = generate_session_id()
    request_id = uuid.uuid4().hex[:8]

    logger.info(f"[{request_id}] Initializing session {session_id}")

    # Generate specs grid
    specs = []
    styles = body.styles or ["none"]
    negation_options = [False, True] if body.include_negations else [False]

    for class_name in body.classes:
        for style in styles:
            for negation in negation_options:
                spec_id = generate_spec_id(class_name, style, negation)
                specs.append({
                    "spec_id": spec_id,
                    "class": class_name,
                    "style": style,
                    "negation": negation
                })

    # Initialize session data with session manager
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": body.classes,
        "target_count_per_class": body.target_count_per_class,
        "styles": styles,
        "include_negations": body.include_negations,
        "specs": specs,
        "samples": [],
        "targets": [
            {"class": class_name, "target_count": body.target_count_per_class}
            for class_name in body.classes
        ]
    }

    if not session_manager.create_session(session_id, session_data):
        raise HTTPException(status_code=500, detail="Failed to create session")

    # Initialize Kumo tables (stub for now)
    try:
        kumo_client = KumoClient()
        # Create empty tables in Kumo and store source tables
        table_metadata = kumo_client.create_session_tables(session_id, specs, session_data["targets"])
        if table_metadata:
            session_manager.update_session(session_id, {"table_metadata": table_metadata})
            logger.info(f"Tables stored in Redis for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to store tables in Redis: {e}")
        # Continue anyway for MVP

    return SessionInitResponse(session_id=session_id, specs=specs)

@app.post("/session/seed_upload", response_model=SeedUploadResponse)
@limiter.limit("20/minute")
async def upload_seed_data_endpoint(
    request: Request,
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    _: str = Depends(verify_token)
) -> SeedUploadResponse:
    """Upload seed data via CSV/JSON file (multipart/form-data)."""
    request_id = uuid.uuid4().hex[:8]

    # File upload path
    if file and session_id:
        # File upload path
        logger.info(f"[{request_id}] Processing file upload for session {session_id}")
        session = get_session(session_id)

        # Read file content
        content = await file.read()

        # Parse based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
            rows = df.to_dict('records')
        elif file.filename.endswith('.json'):
            rows = json.loads(content)
        else:
            raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")
    else:
        raise HTTPException(status_code=400, detail="File and session_id required")

    # Validate row count
    current_count = len(session["samples"])
    new_count = len(rows)

    if current_count + new_count > MAX_TOTAL_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Total rows would exceed maximum of {MAX_TOTAL_ROWS}. Current: {current_count}, Attempted: {new_count}"
        )

    # Normalize and process rows
    processed_rows = []
    for row in rows:
        # Normalize and sanitize fields
        text = sanitize_text(row.get("text", ""))
        class_name = row.get("class", row.get("class_name", ""))
        style = row.get("style", "none")
        negation = bool(row.get("negation", False))

        if not text or not class_name:
            continue

        # Skip rows with unknown classes (just log warning)
        if class_name not in session["classes"]:
            logger.warning(f"Skipping row with unknown class '{class_name}' (not in {session['classes']})")
            continue

        # Generate IDs
        sample_id = generate_sample_id()
        spec_id = generate_spec_id(class_name, style, negation)

        # Create sample record
        sample = {
            "sample_id": sample_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "class": class_name,
            "style": style,
            "negation": negation,
            "source": "seed",
            "spec_id": spec_id
        }

        processed_rows.append(sample)
        session["samples"].append(sample)

    # Update session with new samples
    if processed_rows:
        session_manager.update_session(session_id, {"samples": session["samples"]})

    # Upload to Kumo
    try:
        kumo_client = KumoClient()
        result = kumo_client.add_samples(session_id, processed_rows)
        if result and result.get("stored"):
            logger.info(f"Stored {len(processed_rows)} samples in Redis")
    except Exception as e:
        logger.error(f"Failed to upload to Kumo: {e}")
        # Continue anyway for MVP

    return SeedUploadResponse(
        rows=processed_rows,
        total_rows=len(session["samples"])
    )

@app.post("/session/seed_upload_json", response_model=SeedUploadResponse)
@limiter.limit("20/minute")
async def upload_seed_data_json(
    request: Request,
    body: SeedUploadJSONRequest,
    _: str = Depends(verify_token)
) -> SeedUploadResponse:
    """Upload seed data via JSON body."""
    request_id = uuid.uuid4().hex[:8]
    session_id = body.session_id

    logger.info(f"[{request_id}] Processing JSON upload for session {session_id}")
    session = get_session(session_id)
    rows = [row.dict() for row in body.rows]

    # Validate row count
    current_count = len(session["samples"])
    new_count = len(rows)

    if current_count + new_count > MAX_TOTAL_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Total rows would exceed maximum of {MAX_TOTAL_ROWS}. Current: {current_count}, Attempted: {new_count}"
        )

    # Normalize and process rows
    processed_rows = []
    for row in rows:
        # Normalize and sanitize fields
        text = sanitize_text(row.get("text", ""))
        class_name = row.get("class", row.get("class_name", ""))
        style = row.get("style", "none")
        negation = bool(row.get("negation", False))

        if not text or not class_name:
            continue

        # Skip rows with unknown classes (just log warning)
        if class_name not in session["classes"]:
            logger.warning(f"Skipping row with unknown class '{class_name}' (not in {session['classes']})")
            continue

        # Generate IDs
        sample_id = generate_sample_id()
        spec_id = generate_spec_id(class_name, style, negation)

        # Create sample record
        sample = {
            "sample_id": sample_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "class": class_name,
            "style": style,
            "negation": negation,
            "source": "seed",
            "spec_id": spec_id
        }

        processed_rows.append(sample)
        session["samples"].append(sample)

    # Update session with new samples
    if processed_rows:
        session_manager.update_session(session_id, {"samples": session["samples"]})

    # Upload to Kumo
    try:
        kumo_client = KumoClient()
        result = kumo_client.add_samples(session_id, processed_rows)
        if result and result.get("stored"):
            logger.info(f"Stored {len(processed_rows)} samples in Redis")
    except Exception as e:
        logger.error(f"Failed to upload to Kumo: {e}")
        # Continue anyway for MVP

    return SeedUploadResponse(
        rows=processed_rows,
        total_rows=len(session["samples"])
    )

@app.get("/plan/coverage", response_model=CoverageResponse)
@limiter.limit("30/minute")
async def get_coverage(
    request: Request,
    session_id: str = Query(...),
    _: str = Depends(verify_token)
) -> CoverageResponse:
    """Get predicted coverage per class."""
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Getting coverage for session {session_id}")

    session = get_session(session_id)
    coverage = []

    # Get LocalTables from Redis and build graph once
    kumo_client = KumoClient()
    local_tables = kumo_client.get_local_tables_for_prediction(session_id)
    graph = build_graph(local_tables) if local_tables else None

    for class_name in session["classes"]:
        try:
            # Build PQL and run prediction
            pql = get_coverage_pql(class_name)
            # For KumoRFM, run prediction with LocalGraph
            if graph:
                from app.rfm_predict import run_pql_with_rfm
                pred_count = run_pql_with_rfm(graph, pql)
            else:
                pred_count = 0  # No graph available
        except Exception as e:
            logger.error(f"Failed to get coverage for {class_name}: {e}")
            pred_count = 0  # Stub fallback

        coverage.append({"class": class_name, "pred_count": pred_count})

    return CoverageResponse(coverage=coverage)

@app.get("/plan/specs", response_model=SpecsResponse)
@limiter.limit("30/minute")
async def get_next_specs(
    request: Request,
    session_id: str = Query(...),
    class_name: str = Query(...),
    _: str = Depends(verify_token)
) -> SpecsResponse:
    """Get next recommended specs to generate for a class."""
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Getting specs for session {session_id}, class {class_name}")

    session = get_session(session_id)

    if class_name not in session["classes"]:
        raise HTTPException(status_code=422, detail=f"Class '{class_name}' not in session")

    try:
        # Get LocalTables from Redis and build graph
        kumo_client = KumoClient()
        local_tables = kumo_client.get_local_tables_for_prediction(session_id)
        graph = build_graph(local_tables) if local_tables else None

        # Build PQL and run prediction
        pql = get_specs_pql(class_name)
        if graph:
            from app.rfm_predict import run_pql_with_rfm
            spec_ids = run_pql_with_rfm(graph, pql)
            # Convert result to list if needed
            if not isinstance(spec_ids, list):
                spec_ids = []
        else:
            spec_ids = []  # No graph available
    except Exception as e:
        logger.error(f"Failed to get specs: {e}")
        spec_ids = []  # Stub fallback

    return SpecsResponse(spec_ids=spec_ids)

@app.post("/export/hf", response_model=HFExportResponse)
@limiter.limit("5/minute")
async def export_to_hf(
    request: Request,
    body: HFExportRequest,
    _: str = Depends(verify_token)
) -> HFExportResponse:
    """Export session data to HuggingFace dataset repository."""
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Exporting session {body.session_id} to HF repo {body.repo_id}")

    session = get_session(body.session_id)

    if not session["samples"]:
        raise HTTPException(status_code=422, detail="No samples to export")

    try:
        # Export to HuggingFace
        repo_url = export_to_huggingface(
            session_id=body.session_id,
            repo_id=body.repo_id,
            samples=session["samples"]
        )
    except Exception as e:
        error_response = safe_error_response(e)
        raise HTTPException(status_code=500, detail=error_response["error"])

    return HFExportResponse(repo_url=repo_url)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Vibe Data Director service")

    # Verify environment variables
    required_env = ["KUMO_API_KEY"]
    missing = [var for var in required_env if not os.getenv(var)]
    if missing:
        logger.warning(f"Missing environment variables: {missing}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Vibe Data Director service")

    # Clean up temp files
    try:
        import shutil
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        logger.error(f"Failed to clean up temp directory: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
