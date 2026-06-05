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
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=15),
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


async def translate_query_pico(query: str) -> dict:
    """
    Translate a colloquial or patient-specific query into clean medical keywords
    and inferred conditions using the PICO framework.
    """
    default_res = {
        "pico_analysis": {
            "patient_problem": query,
            "intervention": "",
            "comparison": "",
            "outcome": ""
        },
        "simplified_search_query": query,
        "inferred_diseases": []
    }
    
    settings = get_settings()
    if not settings.mistral_api_key:
        return default_res
        
    try:
        llm = ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=0.0,
            max_tokens=250,
        )
        
        prompt = f"""You are a medical informatics expert. Analyze the raw user clinical query below (which may be conversational, symptom-heavy, or patient-specific) and translate it into search keywords optimized for PubMed, using the PICO (Patient, Intervention, Comparison, Outcome) framework.

User Query:
{query}

Instructions:
1. Identify the Patient/Problem (P), Intervention (I), Comparison (C), and Outcome (O).
2. Translate colloquial symptoms and terms to standard medical subject headings (MeSH) or clinical terms (e.g. "high bp" -> "hypertension", "blurry vision" -> "blurred vision", "heart rate" -> "tachycardia" or "heart rate", "heart attack" -> "myocardial infarction", "high sugar" -> "hyperglycemia" or "diabetes").
3. Generate a simplified search query for PubMed. Keep it simple, keyword-focused, and restricted ONLY to core medical symptoms, clinical conditions, or therapeutic agents (max 3-5 terms, connected by simple spaces, avoiding natural language phrases, symbols like >, <, or question marks). Do NOT include demographics (such as age, gender, e.g., '45 years old', 'middle aged adult', 'male', 'female'), action verbs, or generic clinical words (such as 'management', 'fixes', 'treatment', 'diagnosis', 'options', 'patient', 'clinical', 'care', 'home', 'etiology', 'cause', 'causes', 'symptoms', 'signs'). Keep the query strictly focused on the medical symptoms/diseases.
4. Identify any inferred diseases or conditions (e.g., "hypertension", "blurred vision", "headache"). Do NOT include age, demographics, or generic words as inferred diseases.
5. Output valid JSON only with this structure:
{{
  "pico_analysis": {{
    "patient_problem": "...",
    "intervention": "...",
    "comparison": "...",
    "outcome": "..."
  }},
  "simplified_search_query": "...",
  "inferred_diseases": ["disease1", "disease2", ...]
}}
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
            # Validate output
            if "simplified_search_query" in parsed:
                return parsed
        except Exception:
            pass
            
        return default_res
        
    except Exception as exc:
        logger.error("Error translating query via PICO: %s", exc)
        return default_res

