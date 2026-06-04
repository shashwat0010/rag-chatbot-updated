import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TRUSTED_JOURNALS = {
    "bmj",
    "british medical journal",
    "nature",
    "nature medicine",
    "the lancet",
    "lancet",
    "nejm",
    "new england journal of medicine",
    "jama",
    "annals of internal medicine",
}

_QUESTION_PREFIX = re.compile(
    r"^(what is|what are|how effective|how does|is there|are there|"
    r"does|do|can|should|summarize|summarise|review|evidence for)\s+",
    re.IGNORECASE,
)


def _pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "hi", "hello", "hii", "hey", "howdy", "greetings", "of", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"
}

# Clinical/search filler words that do not represent core medical concepts and should be ignored in keyword fallbacks
FALLBACK_IGNORED = {
    "compare", "comparison", "efficacy", "effectiveness", "side", "effect", 
    "effects", "profile", "profiles", "tolerability", "safety", "treating", 
    "treatment", "treat", "guidelines", "guideline", "managing", "management", 
    "clinical", "trial", "outcomes", "outcome", "patients", "patient", "versus", "vs"
}


def _simplify_query_for_pubmed(query: str) -> str:
    """Turn natural-language questions into PubMed-friendly search terms."""
    q = query.strip().rstrip("?.!")
    
    # Preserve Boolean queries with uppercase AND/OR or parenthesis
    words = q.split()
    if len(words) > 1:
        if any(w in ("AND", "OR") for w in words) or "(" in q or ")" in q:
            return q

    q = _QUESTION_PREFIX.sub("", q)
    # Drop filler phrases common in clinical questions
    for phrase in (
        "the efficacy of ",
        "the effectiveness of ",
        "the role of ",
        "evidence for ",
        "evidence on ",
        "recent ",
        "current ",
    ):
        if q.lower().startswith(phrase):
            q = q[len(phrase) :]
    
    # Final filter: remove common stop words and greetings if they're not the only words
    words = q.split()
    if len(words) > 1:
        filtered = [w for w in words if w.lower() not in _STOP_WORDS]
        if filtered:
            return " ".join(filtered)
            
    return q.strip() or query.strip()


def _make_fallback_query(query: str) -> str:
    """Create a clean, keyword-only fallback query by stripping punctuation, Boolean operators, and stop words."""
    q = query.lower()
    # Replace common punctuation with spaces
    q = re.sub(r"[()\"'?,.!;:\-\[\]]", " ", q)
    
    # Strip question prefixes
    q = _QUESTION_PREFIX.sub("", q)
    
    # Strip clinical filler phrases
    for phrase in (
        "the efficacy of",
        "the effectiveness of",
        "the role of",
        "evidence for",
        "evidence on",
        "recent",
        "current",
    ):
        q = q.replace(phrase, " ")
        
    words = q.split()
    
    # Filter out stop words, Boolean operators, and clinical search fillers
    ignored = _STOP_WORDS.union({"and", "or", "not"}).union(FALLBACK_IGNORED)
    keywords = [w for w in words if w not in ignored]
    
    # Take first 8 unique keywords
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique_keywords.append(w)
            if len(unique_keywords) >= 8:
                break
                
    return " ".join(unique_keywords)


async def _esearch(client: httpx.AsyncClient, term: str, limit: int) -> List[str]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(limit),
        "retmode": "json",
        "sort": "relevance",
        "tool": "medical_research_assistant",
        "email": "research@example.com",
    }
    response = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30.0)
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _fetch_xml(client: httpx.AsyncClient, id_list: List[str]) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "tool": "medical_research_assistant",
        "email": "research@example.com",
    }
    response = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30.0)
    response.raise_for_status()
    return response.text


async def search_pubmed(query: str, max_results: Optional[int] = None) -> List[PaperMetadata]:
    settings = get_settings()
    limit = max_results or settings.pubmed_max_results
    simplified = _simplify_query_for_pubmed(query)

    # Try progressively broader strategies (long questions often fail strict filters)
    search_terms = [
        f"({simplified}) AND (systematic review[pt] OR meta-analysis[pt] OR randomized controlled trial[pt] OR review[pt])",
        simplified,
        _make_fallback_query(query),  # clean keyword-only fallback query
    ]
    # Deduplicate while preserving order
    seen_terms: set[str] = set()
    unique_terms: List[str] = []
    for term in search_terms:
        if term and term not in seen_terms:
            seen_terms.add(term)
            unique_terms.append(term)

    async with httpx.AsyncClient() as client:
        id_list: List[str] = []
        used_term = ""
        for term in unique_terms:
            id_list = await _esearch(client, term, limit)
            if id_list:
                used_term = term
                break
            logger.info("PubMed: no hits for term=%s", term[:100])

        if not id_list:
            logger.info("PubMed returned no results for query: %s", query[:80])
            return []

        logger.info("PubMed: %d hits using term=%s", len(id_list), used_term[:100])
        xml_text = await _fetch_xml(client, id_list)

    return _parse_pubmed_xml(xml_text)


def _parse_year(article: ET.Element) -> Optional[int]:
    for path in (".//PubDate/Year", ".//ArticleDate/Year", ".//DateRevised/Year"):
        el = article.find(path)
        if el is not None and el.text and el.text.isdigit():
            return int(el.text)
    medline = article.find(".//MedlineDate")
    if medline is not None and medline.text:
        match = re.search(r"\d{4}", medline.text)
        if match:
            return int(match.group())
    return None


def _parse_authors(article: ET.Element) -> str:
    names: List[str] = []
    for author in article.findall(".//Author"):
        last = author.findtext("LastName", "")
        fore = author.findtext("ForeName", "")
        if last:
            names.append(f"{fore} {last}".strip() if fore else last)
    return ", ".join(names[:6]) + (" et al." if len(names) > 6 else "")


def _parse_pubmed_xml(xml_text: str) -> List[PaperMetadata]:
    papers: List[PaperMetadata] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse PubMed XML: %s", exc)
        return papers

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()

        title = _clean_text(article.findtext(".//ArticleTitle", ""))
        abstract_parts = [
            _clean_text(el.text)
            for el in article.findall(".//AbstractText")
            if el.text
        ]
        abstract = " ".join(abstract_parts) if abstract_parts else ""

        journal = _clean_text(
            article.findtext(".//Journal/Title")
            or article.findtext(".//MedlineTA")
            or "Unknown Journal"
        )
        year = _parse_year(article)
        authors = _parse_authors(article)
        doi_el = article.find(".//ArticleId[@IdType='doi']")
        doi = doi_el.text if doi_el is not None else None

        if not title:
            continue

        papers.append(
            PaperMetadata(
                pmid=pmid,
                title=title,
                abstract=abstract or "Abstract not available.",
                journal=journal,
                year=year,
                authors=authors or None,
                pubmed_url=_pubmed_url(pmid),
                doi=doi,
            )
        )

    return papers


def prioritize_trusted_journals(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    def score(p: PaperMetadata) -> int:
        j = p.journal.lower()
        if any(t in j for t in TRUSTED_JOURNALS):
            return 0
        return 1

    return sorted(papers, key=score)
