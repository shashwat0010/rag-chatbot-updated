import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    reraise=True
)
async def _call_llm_with_retry(llm: ChatMistralAI, messages: list) -> any:
    """Invoke the LLM with automatic retry on transient errors (like 429 rate limits)."""
    return await llm.ainvoke(messages)

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
class QueryAnalysisResult:
    allowed: bool
    message: Optional[str] = None
    is_emergency: bool = False
    risk_level: str = "LOW"
    category: str = "MEDICAL_IN_SCOPE"
    pico_analysis: dict = None
    simplified_search_query: str = ""
    inferred_diseases: List[str] = None
    synonym_expansion: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.pico_analysis is None:
            self.pico_analysis = {}
        if self.inferred_diseases is None:
            self.inferred_diseases = []
        if self.synonym_expansion is None:
            self.synonym_expansion = {}

# Keep GuardrailResult as alias for backward compatibility/simplicity
GuardrailResult = QueryAnalysisResult


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

async def check_query_safety(query: str, block_emergency: bool = True) -> QueryAnalysisResult:
    normalized = query.strip().lower()
    if len(normalized) < 3:
        return QueryAnalysisResult(
            allowed=False,
            message="Please enter a medical research question (at least 3 characters).",
        )

    # 1. Emergency-related query block (fast regex check)
    if block_emergency:
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning("Emergency-related query blocked (fast regex)")
                return QueryAnalysisResult(
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
        return QueryAnalysisResult(
            allowed=True,
            category="GREETING",
            risk_level="LOW",
            simplified_search_query="",
            inferred_diseases=[],
            synonym_expansion={}
        )

    # 3. LLM Unified Query Analyzer (Scope, Emergency, High-Risk, PICO, Synonyms)
    settings = get_settings()
    if not settings.mistral_api_key:
        return QueryAnalysisResult(
            allowed=True,
            risk_level="LOW",
            category="MEDICAL_IN_SCOPE",
            simplified_search_query=query,
            inferred_diseases=[],
            synonym_expansion={}
        )

    try:
        llm = ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=0.0,
            max_tokens=600,
        )

        prompt = f"""You are a medical query analyzer and safety classifier.
Analyze the user query below and provide safety classifications, clinical intent, a PICO framework breakdown, optimized PubMed search terms, and dynamic synonyms.

Instructions:
1. Classify safety and category:
   - "category": Choose one of:
     - "MEDICAL_IN_SCOPE": Professional medical research questions, queries about clinical trials, outcomes, drug efficacy, medical guidelines, or physiological mechanisms.
     - "PATIENT_SPECIFIC": Queries seeking patient-specific diagnosis, clinical treatment advice, prescribing decisions, or medical management of a specific patient's symptoms (e.g. "What should I prescribe?", "Here is my patient with X, what do I do?").
     - "NON_MEDICAL": General, irrelevant, or non-medical queries (e.g., weather, history, coding).
    - "is_emergency": true ONLY if the query describes a clear, immediate, acute life-threatening medical emergency (such as active chest pain, cardiac arrest, severe bleeding, active stroke, anaphylaxis, or suicide risk). Do NOT flag sub-acute symptoms like a multi-day headache, blurry vision, high blood pressure, or general discomfort as emergencies, even if they require medical attention.
    - "is_high_risk": true if the query suggests replacing critical medical treatments (like chemotherapy, insulin, or prescribed life-saving drugs) with unverified alternative/home remedies, or stopping vital treatments.

2. Identify clinical focus/intent:
   - "clinical_focus": Choose one of: "treatment", "diagnosis", "mechanism_of_action", "prognosis", "general".

3. PICO Analysis:
   - Extract the patient/problem, intervention, comparison, and outcome fields.

4. Normalized & Simplified search keywords:
   - "simplified_search_query": A simple, keyword-focused PubMed search query. Map all colloquial/casual symptoms and terms to standard medical terminology (e.g., "high blood pressure" -> "hypertension", "fast heart rate" -> "tachycardia", "high sugar" -> "hyperglycemia" or "diabetes").
   - Restrict this ONLY to core medical symptoms, clinical conditions, or therapeutic agents (max 3-5 terms separated by spaces).
   - Do NOT include demographics (such as age, gender, e.g. "45 years old"), verbs, or generic clinical words (e.g. "treatment", "management", "diagnosis", "cause", "patient", "what", "is", "considered", "fixes").

5. Inferred conditions:
   - "inferred_diseases": A list of at most 1-2 most likely primary underlying medical conditions/diseases inferred from the patient's symptoms (e.g., if symptoms are polyuria and polydipsia, list ["Type 2 Diabetes"]). Avoid listing broad differential diagnoses or multiple alternative conditions.

6. Synonym expansion:
   - "synonym_expansion": A JSON dictionary mapping each medical keyword in "simplified_search_query" and each inferred disease in "inferred_diseases" to a list of 2-4 alternative medical synonyms or related MeSH terms.

Output valid JSON only with this structure:
{{
  "category": "MEDICAL_IN_SCOPE" | "PATIENT_SPECIFIC" | "NON_MEDICAL",
  "is_emergency": true | false,
  "is_high_risk": true | false,
  "clinical_focus": "treatment" | "diagnosis" | "mechanism_of_action" | "prognosis" | "general",
  "pico_analysis": {{
    "patient_problem": "...",
    "intervention": "...",
    "comparison": "...",
    "outcome": "..."
  }},
  "simplified_search_query": "...",
  "inferred_diseases": ["disease1", "disease2", ...],
  "synonym_expansion": {{
    "term1": ["syn1", "syn2", ...],
    "term2": ["syn1", "syn2", ...]
  }}
}}

Examples:
Query: My blood pressure was 150/95 at home. Is that considered high? What medication usually fixes that?
Response:
{{
  "category": "PATIENT_SPECIFIC",
  "is_emergency": false,
  "is_high_risk": false,
  "clinical_focus": "treatment",
  "pico_analysis": {{
    "patient_problem": "high blood pressure (150/95)",
    "intervention": "antihypertensive medication",
    "comparison": "none",
    "outcome": "blood pressure control"
  }},
  "simplified_search_query": "hypertension",
  "inferred_diseases": ["Hypertension"],
  "synonym_expansion": {{
    "hypertension": ["hypertension", "high blood pressure", "elevated blood pressure"],
    "Hypertension": ["hypertension", "high blood pressure", "elevated blood pressure"]
  }}
}}

Query: I've had a headache for three days and my vision is a bit blurry. Is this something I need to worry about?
Response:
{{
  "category": "PATIENT_SPECIFIC",
  "is_emergency": false,
  "is_high_risk": false,
  "clinical_focus": "diagnosis",
  "pico_analysis": {{
    "patient_problem": "persistent headache and blurred vision",
    "intervention": "none",
    "comparison": "none",
    "outcome": "diagnosis or risk assessment"
  }},
  "simplified_search_query": "headache blurred vision persistent",
  "inferred_diseases": ["Migraine with aura", "Idiopathic intracranial hypertension"],
  "synonym_expansion": {{
    "headache": ["headache", "cephalalgia", "head pain", "migraine"],
    "blurred vision": ["blurred vision", "visual disturbance", "blurry vision"],
    "persistent": ["persistent", "prolonged", "chronic"],
    "Migraine with aura": ["Migraine with aura", "classic migraine", "migraine with visual aura"],
    "Idiopathic intracranial hypertension": ["Idiopathic intracranial hypertension", "IIH", "pseudotumor cerebri"]
  }}
}}

Query: i am 45 years old, i am having high bp and heart rate
Response:
{{
  "category": "PATIENT_SPECIFIC",
  "is_emergency": false,
  "is_high_risk": false,
  "clinical_focus": "diagnosis",
  "pico_analysis": {{
    "patient_problem": "high blood pressure and fast heart rate in a 45-year-old",
    "intervention": "none",
    "comparison": "none",
    "outcome": "diagnosis or investigation"
  }},
  "simplified_search_query": "hypertension tachycardia",
  "inferred_diseases": ["Hypertension", "Tachycardia"],
  "synonym_expansion": {{
    "hypertension": ["hypertension", "high blood pressure", "elevated blood pressure"],
    "tachycardia": ["tachycardia", "fast heart rate", "rapid heartbeat"],
    "Hypertension": ["hypertension", "high blood pressure", "elevated blood pressure"],
    "Tachycardia": ["tachycardia", "fast heart rate", "rapid heartbeat"]
  }}
}}

User Query:
{query}
"""
        messages = [HumanMessage(content=prompt)]
        response = await _call_llm_with_retry(llm, messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = {}
        try:
            parsed = json.loads(content)
        except Exception as json_exc:
            logger.error("Failed to parse LLM query analysis JSON: %s. Content: %s", json_exc, content)
            # Try parsing key elements using regex fallback
            category_match = re.search(r'"category"\s*:\s*"([^"]+)"', content)
            category = category_match.group(1).upper() if category_match else "MEDICAL_IN_SCOPE"
            is_emergency = "true" in re.search(r'"is_emergency"\s*:\s*(\w+)', content).group(1).lower() if re.search(r'"is_emergency"\s*:\s*(\w+)', content) else False
            is_high_risk = "true" in re.search(r'"is_high_risk"\s*:\s*(\w+)', content).group(1).lower() if re.search(r'"is_high_risk"\s*:\s*(\w+)', content) else False
            simplified_query_match = re.search(r'"simplified_search_query"\s*:\s*"([^"]+)"', content)
            simplified_query = simplified_query_match.group(1) if simplified_query_match else query
            parsed = {
                "category": category,
                "is_emergency": is_emergency,
                "is_high_risk": is_high_risk,
                "simplified_search_query": simplified_query,
                "inferred_diseases": [],
                "synonym_expansion": {}
            }

        category = parsed.get("category", "MEDICAL_IN_SCOPE").upper()
        is_emergency = bool(parsed.get("is_emergency", False))
        is_high_risk = bool(parsed.get("is_high_risk", False))
        simplified_query = parsed.get("simplified_search_query", query)
        inferred_diseases = parsed.get("inferred_diseases", [])
        synonym_expansion = parsed.get("synonym_expansion", {})
        pico_analysis = parsed.get("pico_analysis", {})

        allowed = True
        message = None
        risk_level = "LOW"

        if is_emergency:
            allowed = False
            risk_level = "EMERGENCY"
            message = (
                "This appears to describe an acute medical emergency. "
                "Do not use this chatbot for urgent care. Seek immediate "
                "emergency medical attention or call your local emergency number."
            )
        elif category == "NON_MEDICAL":
            allowed = False
            risk_level = "NON_MEDICAL"
            message = "I am designed to assist with medical literature and clinical evidence. I cannot answer queries outside of this scope."
        elif category == "PATIENT_SPECIFIC":
            allowed = True
            risk_level = "PATIENT_SPECIFIC"
        else: # MEDICAL_IN_SCOPE
            if is_high_risk:
                allowed = True
                risk_level = "HIGH"
                message = "High-risk medical intent detected."

        # Keep manual regex override for high-risk pattern safety check
        if allowed and risk_level != "HIGH":
            for pattern in HIGH_RISK_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    risk_level = "HIGH"
                    message = "High-risk medical intent detected."
                    break

        logger.info("Dynamic Query classification: category=%s | risk=%s | allowed=%s", category, risk_level, allowed)

        return QueryAnalysisResult(
            allowed=allowed,
            message=message,
            is_emergency=is_emergency,
            risk_level=risk_level,
            category=category,
            pico_analysis=pico_analysis,
            simplified_search_query=simplified_query,
            inferred_diseases=inferred_diseases,
            synonym_expansion=synonym_expansion
        )

    except Exception as exc:
        logger.error("Error analyzing query in LLM: %s", exc)
        # Fallback to pattern-based classification if LLM fails
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning("High-risk medical intent detected (fallback)")
                return QueryAnalysisResult(
                    allowed=True,
                    risk_level="HIGH",
                    message="High-risk medical intent detected.",
                    simplified_search_query=query,
                )

        treatment_only = re.search(
            r"^(should i take|what dose should i|prescribe me|treat my|what should i prescribe|diagnose this patient|what do i prescribe)\b",
            normalized,
        )
        if treatment_only:
            return QueryAnalysisResult(
                allowed=True,
                risk_level="PATIENT_SPECIFIC",
                message=None,
                simplified_search_query=query,
            )

        return QueryAnalysisResult(
            allowed=True,
            risk_level="LOW",
            message=None,
            simplified_search_query=query,
        )


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
