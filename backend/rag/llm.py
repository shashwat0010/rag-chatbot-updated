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


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    reraise=True
)
async def _get_llm_stream_with_retry(llm: ChatMistralAI, messages: list):
    """Call astream and wait for the first chunk with retry to handle initial connection 429s."""
    stream = llm.astream(messages)
    try:
        iterator = stream.__aiter__()
        first_chunk = await iterator.__anext__()
    except StopAsyncIteration:
        async def empty_gen():
            if False:
                yield None
        return empty_gen()
    
    async def yield_all():
        yield first_chunk
        async for chunk in iterator:
            yield chunk

    return yield_all()


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
- If the query asks about drug dosages, specific clinical parameters, precautions, or specific criteria, you must extract and provide the exact values, numbers, guidelines, and dosages from the text rather than a general summary. Do not synthesize or omit specific quantitative clinical details (e.g., '10-20 mg daily', 'eGFR < 30 ml/min/1.73m2').
- Set insufficient_evidence to true and use empty arrays ONLY if there is absolutely no relevant data in the context. If the context has papers about the topic (e.g., curcumin and cancer) but they do not show a definitive cure, do NOT set insufficient_evidence to true; instead, summarize the studied effects of the compound cautiously and explicitly note that clinical evidence for a cure is lacking.
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

STREAM_SYSTEM_PROMPT = """You are a medical research assistant for licensed clinicians.
Answer ONLY from the provided PubMed abstract excerpts.

If the user query is a greeting like 'hi' or 'hello', or asks who you are, provide a friendly professional introduction about your capabilities.

Formatting rules (strict):
- Use Markdown structure.
- Output a section starting with **Summary:** containing a one-sentence takeaway (max 35 words).
- Output a section starting with **Key findings:** followed by bullet points (one finding per line). Put each distinct finding on its own bullet. Include citation markers like [1], [2] on each bullet.
- Output an optional section starting with **Clinical notes:** followed by bullet points for limitations or cautions.
- Do NOT output JSON, HTML, or code blocks. Output Markdown directly.
- If the sources contain relevant research about the concepts but do not support a definitive answer to the user's specific request (e.g., if they ask about a 'cure' but the literature only shows lab activity), do not write the default insufficient evidence phrase. Instead, summarize the studied effects of the compound cautiously, explaining that while lab activity is observed, clinical evidence for a cure is lacking.
- If the query asks about drug dosages, specific clinical parameters, precautions, or specific criteria, you must extract and provide the exact values, numbers, guidelines, and dosages from the text rather than a general summary. Do not synthesize or omit specific quantitative clinical details (e.g., '10-20 mg daily', 'eGFR < 30 ml/min/1.73m2').
- Use cautious language (may, suggests, limited evidence).
- No personal medical advice or emergency instructions.
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

    async def generate_stream(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ):
        if is_greeting_or_meta(query) and not chunks:
            import asyncio
            parsed = json.loads(GREETING_RESPONSE)
            text = _build_answer_from_parsed(parsed)
            # Yield in smaller chunks to simulate streaming for a better UI experience
            words = text.split(" ")
            for w in words:
                yield w + " "
                await asyncio.sleep(0.02)
            return

        if not chunks:
            yield INSUFFICIENT_EVIDENCE_MESSAGE
            return

        context = _format_context(chunks)
        user_prompt = f"""Question: {query}

Sources:
{context}

Respond in Markdown format (Summary, Key findings, and Clinical notes sections with bullet points). Include citations [1], [2], etc. directly on findings."""

        messages = [
            SystemMessage(content=STREAM_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        stream = await _get_llm_stream_with_retry(self._llm, messages)
        async for chunk in stream:
            yield chunk.content
