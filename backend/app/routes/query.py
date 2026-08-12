import logging
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
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


from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session
from models.database import SearchHistory, get_db, User
from services.auth import get_current_user


@router.post("/query", response_model=QueryResponse)
@router.post("/api/query", response_model=QueryResponse)
@limiter.limit(_rate_limit)
async def query_medical_research(
    request: Request,
    body: QueryRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> QueryResponse:
    settings = get_settings()
    api_key = settings.openrouter_api_key or settings.mistral_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured. Set OPENROUTER_API_KEY or MISTRAL_API_KEY in backend/.env",
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
        result = await get_pipeline().run(
            body.query,
            max_papers=body.max_papers,
            risk_level=safety.risk_level,
            analysis=safety
        )
        if not result.confidence_note.startswith("Low confidence"):
            result.confidence_note = f"{result.confidence_note} {DISCLAIMER}"

        # Persist query & response to search history if authenticated user
        if current_user and db:
            try:
                history_entry = SearchHistory(
                    user_id=current_user.id,
                    query=body.query,
                    response=result.answer,
                    confidence_score=f"{result.confidence_score:.2f}",
                    evidence_count=len(result.citations)
                )
                db.add(history_entry)
                db.commit()
            except Exception as e:
                logger.error("Failed to save search history: %s", e)

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


@router.post("/query/stream")
@router.post("/api/query/stream")
@limiter.limit(_rate_limit)
async def query_medical_research_stream(
    request: Request,
    body: QueryRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    settings = get_settings()
    api_key = settings.openrouter_api_key or settings.mistral_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured. Set OPENROUTER_API_KEY or MISTRAL_API_KEY in backend/.env",
        )

    safety = await check_query_safety(body.query, settings.block_emergency_keywords)
    if not safety.allowed:
        if safety.risk_level in ("NON_MEDICAL", "PATIENT_SPECIFIC"):
            async def error_generator():
                refusal_msg = safety.message
                yield f"event: token\ndata: {json.dumps(refusal_msg)}\n\n"
                metadata = {
                    "citations": [],
                    "confidence_note": f"Policy Check: {safety.message}",
                    "confidence_score": 0.0,
                    "insufficient_evidence": False,
                    "sources_searched": [],
                    "confidence_label": "Scope Refusal"
                }
                yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")
            
        if not is_greeting_or_meta(body.query):
            raise HTTPException(status_code=400, detail=safety.message)

    logger.info("Streaming query from %s: %s", request.client.host if request.client else "unknown", body.query[:100])

    try:
        pipeline = get_pipeline()
        raw_generator = pipeline.run_stream(
            body.query,
            max_papers=body.max_papers,
            risk_level=safety.risk_level,
            analysis=safety
        )

        async def history_saving_generator():
            full_response_chunks = []
            confidence_score_val = "0.00"
            evidence_count_val = 0

            async for chunk in raw_generator:
                yield chunk
                try:
                    if chunk.startswith("event: token\ndata: "):
                        token_data = chunk.split("data: ", 1)[1].strip()
                        token_str = json.loads(token_data)
                        full_response_chunks.append(token_str)
                    elif chunk.startswith("event: metadata\ndata: "):
                        meta_data = chunk.split("data: ", 1)[1].strip()
                        meta_dict = json.loads(meta_data)
                        if "confidence_score" in meta_dict:
                            confidence_score_val = f"{meta_dict['confidence_score']:.2f}"
                        if "citations" in meta_dict:
                            evidence_count_val = len(meta_dict["citations"])
                except Exception:
                    pass

            if current_user and db:
                try:
                    accumulated_response = "".join(full_response_chunks).strip()
                    if accumulated_response:
                        history_entry = SearchHistory(
                            user_id=current_user.id,
                            query=body.query,
                            response=accumulated_response,
                            confidence_score=confidence_score_val,
                            evidence_count=evidence_count_val
                        )
                        db.add(history_entry)
                        db.commit()
                        logger.info("Saved search history entry for user %s (query: %s)", current_user.id, body.query[:50])
                except Exception as e:
                    logger.error("Failed to save search history in stream: %s", e)

        return StreamingResponse(history_saving_generator(), media_type="text/event-stream")
    except Exception as exc:
        logger.exception("Streaming query processing failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your streaming research query. Please try again.",
        ) from exc
