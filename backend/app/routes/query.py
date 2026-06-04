import logging

from fastapi import APIRouter, HTTPException, Request
import httpx

from app.config import get_settings
from app.limiter import limiter
from models.schemas import QueryRequest, QueryResponse
from rag.pipeline import RAGPipeline
from services.embeddings import EmbeddingServiceError, OpenAIQuotaError
from services.guardrails import DISCLAIMER, check_query_safety, is_greeting_or_meta

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])
_rate_limit = f"{get_settings().rate_limit_per_minute}/minute"

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _llm_quota_message() -> str:
    return (
        "LLM API quota exceeded. Please check your Mistral billing status. "
        "You can also set USE_LOCAL_EMBEDDINGS=true in backend/.env to use free local search."
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit(_rate_limit)
async def query_medical_research(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    settings = get_settings()
    if not settings.mistral_api_key:
        raise HTTPException(
            status_code=503,
            detail="Mistral API key is not configured. Set MISTRAL_API_KEY in the environment.",
        )

    safety = await check_query_safety(body.query, settings.block_emergency_keywords)
    if not safety.allowed:
        if safety.risk_level in ("NON_MEDICAL", "PATIENT_SPECIFIC"):
            return QueryResponse(
                answer=safety.message,
                citations=[],
                confidence_note=f"Policy Check: {safety.message}",
                confidence_score=0.0,
                insufficient_evidence=False,
                sources_searched=[],
                confidence_label="Scope Refusal"
            )
        # If it's too short, but it's a greeting, we might want to allow it
        if not is_greeting_or_meta(body.query):
            raise HTTPException(status_code=400, detail=safety.message)

    logger.info("Query from %s: %s", request.client.host if request.client else "unknown", body.query[:100])

    try:
        result = await get_pipeline().run(body.query, max_papers=body.max_papers, risk_level=safety.risk_level)
        if not result.confidence_note.startswith("Low confidence"):
            result.confidence_note = f"{result.confidence_note} {DISCLAIMER}"
        return result
    except OpenAIQuotaError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail="API rate limit reached. Please wait a moment and try again.",
            ) from exc
        if status == 401:
            raise HTTPException(
                status_code=401,
                detail="Invalid Mistral API key. Check MISTRAL_API_KEY in backend/.env",
            ) from exc
        raise HTTPException(status_code=502, detail="LLM API Error") from exc
    except (EmbeddingServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Configuration error during query")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your research query. Please try again.",
        ) from exc
