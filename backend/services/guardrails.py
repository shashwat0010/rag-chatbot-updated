import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

from app.config import get_settings

logger = logging.getLogger(__name__)

EMERGENCY_PATTERNS = [
    r"\b(chest pain|heart attack|stroke|suicid|overdose|can't breathe|cannot breathe)\b",
    r"\b(severe bleeding|unconscious|anaphylaxis|cardiac arrest)\b",
    r"\b(911|emergency room now|dying)\b",
]

HIGH_RISK_PATTERNS = [
    r"\b(instead of chemotherapy|stop insulin|stop medication|replace chemotherapy|avoid professional treatment|home remedies instead of)\b",
    r"\b(treat chest pain at home|self-diagnosis|can i stop)\b"
]

DISCLAIMER = (
    "This tool provides research summaries for licensed clinicians and does not "
    "replace clinical judgment, diagnosis, or emergency care."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Current evidence is insufficient to provide a reliable answer."
)


@dataclass
class GuardrailResult:
    allowed: bool
    message: Optional[str] = None
    is_emergency: bool = False
    risk_level: str = "LOW"


GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening|hii|hii+)(\s.*)?$",
    r"^(who are you|what can you do|how can you help|help)$",
]

def is_greeting_or_meta(query: str) -> bool:
    normalized = query.strip().lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False

async def check_query_safety(query: str, block_emergency: bool = True) -> GuardrailResult:
    normalized = query.strip().lower()
    if len(normalized) < 3:
        return GuardrailResult(
            allowed=False,
            message="Please enter a medical research question (at least 3 characters).",
        )

    # 1. Emergency-related query block (fast regex check)
    if block_emergency:
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning("Emergency-related query blocked")
                return GuardrailResult(
                    allowed=False,
                    is_emergency=True,
                    risk_level="EMERGENCY",
                    message=(
                        "This appears to describe an acute medical emergency. "
                        "Do not use this chatbot for urgent care. Seek immediate "
                        "emergency medical attention or call your local emergency number."
                    ),
                )

    # 2. Greeting check (fast regex check)
    if is_greeting_or_meta(query):
        return GuardrailResult(allowed=True)

    # 3. LLM Query Safety Guardrail (Scope and Refusals)
    settings = get_settings()
    if not settings.mistral_api_key:
        # Fallback if API key is missing
        return GuardrailResult(allowed=True)

    try:
        llm = ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=0.0,
            max_tokens=100,
        )

        prompt = f"""You are a medical scope classifier.
Analyze the user query below and classify it into one of these four categories:
- "GREETING": Greetings, salutations, or questions asking who you are or what you do.
- "MEDICAL_IN_SCOPE": Professional medical research questions, queries about clinical trials/outcomes/efficacy, medical guidelines (e.g., society guidelines), or drug/medical conceptual mechanisms of action.
- "PATIENT_SPECIFIC": Queries seeking patient-specific diagnosis, clinical treatment advice, prescribing decisions, or medical management of a specific patient's symptoms (e.g., "What should I prescribe for this patient?", "Diagnose this patient", "Here is a patient with X, what do I do?").
- "NON_MEDICAL": General, irrelevant, or non-medical queries (e.g., weather, stocks, general opinions, history, math, coding, etc.).

Query: {query}

Output valid JSON only with this structure:
{{
  "category": "GREETING" | "MEDICAL_IN_SCOPE" | "PATIENT_SPECIFIC" | "NON_MEDICAL",
  "reason": "Brief explanation"
}}
"""
        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            category = parsed.get("category", "").upper()
        except Exception:
            match = re.search(r'"category"\s*:\s*"([^"]+)"', content)
            if match:
                category = match.group(1).upper()
            else:
                category = "MEDICAL_IN_SCOPE"

        logger.info("Query classification: %s for query: %s", category, query[:100])

        if category == "NON_MEDICAL":
            return GuardrailResult(
                allowed=False,
                risk_level="NON_MEDICAL",
                message="I am designed to assist with medical literature and clinical evidence. I cannot answer queries outside of this scope."
            )
        elif category == "PATIENT_SPECIFIC":
            return GuardrailResult(
                allowed=True,
                risk_level="PATIENT_SPECIFIC",
                message=None
            )
        elif category == "GREETING":
            return GuardrailResult(allowed=True)
        else:  # MEDICAL_IN_SCOPE
            # Check high risk patterns to set risk_level to HIGH (e.g. replacing chemo)
            for pattern in HIGH_RISK_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    logger.warning("High-risk medical intent detected")
                    return GuardrailResult(
                        allowed=True,
                        risk_level="HIGH",
                        message="High-risk medical intent detected."
                    )
            return GuardrailResult(allowed=True)

    except Exception as exc:
        logger.error("Error classifying query: %s", exc)
        # Fallback to pattern-based classification if LLM fails
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning("High-risk medical intent detected (fallback)")
                return GuardrailResult(
                    allowed=True,
                    risk_level="HIGH",
                    message="High-risk medical intent detected."
                )

        treatment_only = re.search(
            r"^(should i take|what dose should i|prescribe me|treat my|what should i prescribe|diagnose this patient|what do i prescribe)\b",
            normalized,
        )
        if treatment_only:
            return GuardrailResult(
                allowed=True,
                risk_level="PATIENT_SPECIFIC",
                message=None
            )

        return GuardrailResult(allowed=True)


def validate_answer_grounding(
    answer: str,
    llm_insufficient: bool,
    confidence_score: float,
) -> Tuple[str, bool, str]:
    """Only reject answers when the model explicitly flags insufficient evidence."""
    if llm_insufficient:
        return INSUFFICIENT_EVIDENCE_MESSAGE, True, (
            "The retrieved abstracts did not contain enough detail for a specific conclusion. "
            "Review cited papers directly."
        )
    note = _confidence_note(confidence_score)
    return answer, False, note


def _confidence_note(score: float) -> str:
    if score >= 0.72:
        return (
            "Good alignment between retrieved abstracts and your question. "
            "Verify against full-text sources and current guidelines."
        )
    if score >= 0.55:
        return (
            "Moderate confidence based on retrieved literature. "
            "Interpret findings cautiously and consider study quality."
        )
    return (
        "Evidence was retrieved but semantic match scores were modest. "
        "Review cited abstracts and consider additional literature search."
    )
