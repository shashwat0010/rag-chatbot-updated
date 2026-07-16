import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.limiter import limiter
from app.concurrency import request_id_var
from app.logging_config import setup_logging
from app.routes import health, query, search, whatsapp
import psutil
import logging

logger = logging.getLogger(__name__)

setup_logging()
# Force uvicorn reload to load updated environment settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application lifespan...")
    mem_before = psutil.virtual_memory().used / (1024 * 1024)
    logger.info("Base RAM before API starts: %.1f MB", mem_before)
    
    yield


app = FastAPI(
    title="Medical Research Assistant API",
    description=(
        "Evidence-grounded medical research API using PubMed retrieval and RAG. "
        "For clinician research support only — not for patient diagnosis or emergency care."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Medical Research Assistant API is running"}


app.include_router(health.router)
app.include_router(query.router)
app.include_router(search.router)
app.include_router(whatsapp.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
