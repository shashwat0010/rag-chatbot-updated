from fastapi import APIRouter

from app.config import get_settings
from models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        llm_configured=bool(settings.openrouter_api_key or settings.mistral_api_key),
    )

