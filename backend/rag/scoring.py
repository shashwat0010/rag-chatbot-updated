"""Calibrate retrieval scores into clinician-facing confidence (0–1)."""

from typing import List

from models.schemas import PaperMetadata
from rag.vector_store import RetrievedChunk
from services.pubmed import TRUSTED_JOURNALS


def _has_trusted_journal(papers: List[PaperMetadata]) -> bool:
    for p in papers[:5]:
        j = p.journal.lower()
        if any(t in j for t in TRUSTED_JOURNALS):
            return True
    return False


def _calibrate_raw_score(raw: float, use_local: bool) -> float:
    """Map embedding similarity to a comparable confidence scale."""
    if use_local:
        # TF-IDF cosine similarity is typically lower than OpenAI embeddings
        if raw <= 0.08:
            return raw
        return min(0.92, 0.42 + (raw - 0.08) * 1.35)
    
    # Mistral cosine similarities are compressed in the high range (typically 0.65 to 0.95)
    if raw <= 0.70:
        return round(max(0.0, raw * 0.2), 3)
    if raw <= 0.85:
        return round(0.14 + (raw - 0.70) * 3.0, 3)
    return round(min(0.95, 0.59 + (raw - 0.85) * 3.6), 3)


def compute_retrieval_confidence(
    chunks: List[RetrievedChunk],
    papers: List[PaperMetadata],
    *,
    use_local: bool,
    query_quality_score: float = 1.0,
    risk_level: str = "LOW",
    has_inferred_disease: bool = True,
) -> float:
    if not chunks:
        return 0.0

    top = max(c.score for c in chunks)
    top3 = sorted((c.score for c in chunks), reverse=True)[:3]
    avg_top3 = sum(top3) / len(top3)

    base = _calibrate_raw_score(top, use_local) * 0.6 + _calibrate_raw_score(avg_top3, use_local) * 0.4
    
    # Penalties for poor query or high risk
    if query_quality_score < 1.0:
        base *= max(0.4, query_quality_score)
        
    if not has_inferred_disease and top < 0.6:
        base *= 0.85
        
    if risk_level == "HIGH":
        base *= 0.5

    # Boost when PubMed returned a solid set of papers
    paper_bonus = min(0.12, len(papers) * 0.008)
    if _has_trusted_journal(papers):
        paper_bonus += 0.06

    return round(min(0.95, base + paper_bonus), 3)


def retrieval_is_sufficient(
    chunks: List[RetrievedChunk],
    papers: List[PaperMetadata],
    *,
    use_local: bool,
    min_papers: int = 2,
) -> bool:
    if len(papers) < 1 or not chunks:
        return False
    if len(papers) >= min_papers and len(chunks) >= 1:
        top = max(c.score for c in chunks)
        floor = 0.12 if use_local else 0.78
        return top >= floor
    return False
