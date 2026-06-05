import json
import logging
import re
from typing import List, Tuple

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from rag.formatting import format_structured_answer, paragraph_to_bullets
from rag.vector_store import RetrievedChunk
from services.guardrails import INSUFFICIENT_EVIDENCE_MESSAGE, is_greeting_or_meta

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    reraise=True
)
async def _call_llm_with_retry(llm: ChatMistralAI, messages: list) -> any:
    """Invoke the LLM with automatic retry on transient errors (like 429 rate limits)."""
    return await llm.ainvoke(messages)


GREETING_RESPONSE = """{
  "summary": "Hello! I am your Medical Research Assistant.",
  "key_findings": [
    "I can help you search PubMed for peer-reviewed literature and RCTs.",
    "I synthesize evidence into structured research summaries with citations.",
    "Please ask a clinical or research-related question to begin."
  ],
  "clinical_notes": [
    "I am for research support only and not for patient diagnosis or emergency care."
  ],
  "cited_indices": [],
  "insufficient_evidence": false
}"""

SYSTEM_PROMPT = """You are a medical research assistant for licensed clinicians.
Answer ONLY from the provided PubMed abstract excerpts.

If the user query is a greeting like 'hi' or 'hello', or asks who you are, provide a friendly professional introduction about your capabilities as a medical research assistant.

Formatting rules (strict):
- Do NOT write long paragraphs or essay-style prose.
- Use short structured sections only.
- Put each distinct finding on its own bullet (one idea per line).
- Include citation markers [1], [2] on each bullet that uses that source.

Content rules:
- Never invent studies, statistics, or recommendations not in the context.
- If context is inadequate, set insufficient_evidence to true and use empty arrays.
- Use cautious language (may, suggests, limited evidence).
- No personal medical advice or emergency instructions.

Output valid JSON only:
{
  "summary": "One sentence takeaway (max 35 words)",
  "key_findings": [
    "Bullet finding with citation [1]",
    "Another bullet [2]"
  ],
  "clinical_notes": [
    "Optional limitation or caution [1]"
  ],
  "cited_indices": [1, 2],
  "insufficient_evidence": false
}
"""


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        p = chunk.paper
        lines.append(
            f"[{i}] PMID:{p.pmid} | {p.title} | {p.journal} ({p.year or 'n/a'})\n"
            f"Relevance:{chunk.score:.3f}\n{chunk.text}\n"
        )
    return "\n".join(lines)


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _build_answer_from_parsed(parsed: dict) -> str:
    if parsed.get("insufficient_evidence"):
        return INSUFFICIENT_EVIDENCE_MESSAGE

    summary = parsed.get("summary") or ""
    key_findings = parsed.get("key_findings") or []
    clinical_notes = parsed.get("clinical_notes") or []

    # Legacy single "answer" field support
    if not key_findings and parsed.get("answer"):
        return paragraph_to_bullets(str(parsed["answer"]))

    if isinstance(key_findings, str):
        key_findings = [key_findings]
    if isinstance(clinical_notes, str):
        clinical_notes = [clinical_notes]

    formatted = format_structured_answer(
        summary=summary if summary else None,
        key_findings=list(key_findings),
        clinical_notes=list(clinical_notes) if clinical_notes else None,
    )
    return formatted or INSUFFICIENT_EVIDENCE_MESSAGE


class MedicalLLM:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is not configured")
        self._llm = ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=0.1,
            max_tokens=900,
        )

    async def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[str, List[int], bool]:
        # Handle greetings immediately without LLM if possible, 
        # or guide the LLM to handle it via prompt.
        if is_greeting_or_meta(query) and not chunks:
            try:
                parsed = json.loads(GREETING_RESPONSE)
                return _build_answer_from_parsed(parsed), [], False
            except Exception:
                pass

        if not chunks:
            return INSUFFICIENT_EVIDENCE_MESSAGE, [], True

        context = _format_context(chunks)
        user_prompt = f"""Question: {query}

Sources:
{context}

Respond with structured JSON only (bullets, no paragraphs)."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = await _call_llm_with_retry(self._llm, messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = _parse_llm_json(raw)
            insufficient = bool(parsed.get("insufficient_evidence", False))
            answer = _build_answer_from_parsed(parsed)
            cited = parsed.get("cited_indices", [])
            if INSUFFICIENT_EVIDENCE_MESSAGE.lower() in answer.lower():
                insufficient = True
            return answer, cited, insufficient
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to parse LLM JSON: %s | raw=%s", exc, raw[:200])
            if "insufficient" in raw.lower():
                return INSUFFICIENT_EVIDENCE_MESSAGE, [], True
            return paragraph_to_bullets(raw[:800]), [], False
