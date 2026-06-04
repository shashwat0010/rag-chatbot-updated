import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.limiter import limiter
from models.schemas import SearchPapersRequest, SearchPapersResponse
from services.guardrails import check_query_safety
from services.pubmed import prioritize_trusted_journals, search_pubmed

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search"])
_rate_limit = f"{get_settings().rate_limit_per_minute}/minute"


@router.post("/search-papers", response_model=SearchPapersResponse)
@limiter.limit(_rate_limit)
async def search_papers(
    request: Request,
    body: SearchPapersRequest,
) -> SearchPapersResponse:
    safety = await check_query_safety(body.query, block_emergency=False)
    if not safety.allowed:
        raise HTTPException(status_code=400, detail=safety.message)

    logger.info(
        "Paper search from %s: %s",
        request.client.host if request.client else "unknown",
        body.query[:100],
    )

    try:
        papers = await search_pubmed(body.query, max_results=body.max_results)
        papers = prioritize_trusted_journals(papers)
        return SearchPapersResponse(
            papers=papers,
            total=len(papers),
            query=body.query,
        )
    except Exception as exc:
        logger.exception("Paper search failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to search PubMed. Please try again later.",
        ) from exc
