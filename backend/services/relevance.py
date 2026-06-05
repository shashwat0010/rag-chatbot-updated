import json
import logging
import re
from typing import Tuple, Dict, List

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=8),
    reraise=True
)
async def _call_llm_with_retry(llm: ChatMistralAI, messages: list) -> any:
    """Invoke the LLM with automatic retry on transient errors (like 429 rate limits)."""
    return await llm.ainvoke(messages)


async def check_papers_relevance_batch(query: str, papers: list) -> Dict[str, Tuple[bool, str]]:
    """
    Check if the retrieved papers are strictly relevant to the clinical question,
    addressing all core concepts and their primary relationship, in a single batch.
    Returns a dictionary mapping pmid to a tuple of (relevant: bool, reason: str).
    """
    relevance_map = {}
    for p in papers:
        relevance_map[str(p.pmid)] = (True, "Defaulting to relevant.")

    settings = get_settings()
    if not settings.mistral_api_key:
        return relevance_map

    if not papers:
        return relevance_map

    try:
        llm = ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=0.0,
            max_tokens=600,
        )

        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"\n--- Paper {i} ---\nPMID: {p.pmid}\nTitle: {p.title}\nAbstract: {p.abstract}\n"

        prompt = f"""Analyze the retrieved medical papers below and determine if they are directly relevant to the user query.

Question:
{query}

Retrieved Papers:
{papers_text}

Rules:
1. For each paper, determine whether it directly addresses ALL major concepts in the question.
2. If the question contains multiple concepts (e.g., comparing SSRIs vs SNRIs, or obesity and sleep apnea), the paper must address the primary relationship/interaction between them.
3. Partial matches are not enough. Discussing only one concept is NOT enough; the paper must be rejected.
4. Output valid JSON only with this structure:
{{
  "results": [
    {{
      "pmid": "string",
      "relevant": true | false,
      "reason": "Brief explanation referencing concepts matched or missed"
    }}
  ]
}}
Ensure EVERY paper is included in the "results" array with its correct "pmid".
"""
        messages = [HumanMessage(content=prompt)]
        response = await _call_llm_with_retry(llm, messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            results = parsed.get("results", [])
            for item in results:
                pmid = str(item.get("pmid", ""))
                relevant = bool(item.get("relevant", False))
                reason = str(item.get("reason", "No reason provided."))
                relevance_map[pmid] = (relevant, reason)
        except Exception as parse_exc:
            logger.error("Failed to parse batch relevance JSON: %s. Raw content: %s", parse_exc, content)

        for pmid, (relevant, reason) in relevance_map.items():
            logger.info("Batch relevance check for PMID %s: relevant=%s | reason: %s", pmid, relevant, reason)

        return relevance_map

    except Exception as exc:
        logger.error("Error checking paper relevance batch after retries: %s", exc)
        return relevance_map


async def check_paper_relevance(query: str, title: str, abstract: str) -> Tuple[bool, str]:
    """
    Check if a single retrieved paper is strictly relevant to the clinical question.
    (Kept for compatibility with unit tests).
    """
    class DummyPaper:
        def __init__(self, pmid, title, abstract):
            self.pmid = pmid
            self.title = title
            self.abstract = abstract

    paper = DummyPaper("dummy", title, abstract)
    results = await check_papers_relevance_batch(query, [paper])
    return results.get("dummy", (True, "Fallback due to missing result in batch."))
