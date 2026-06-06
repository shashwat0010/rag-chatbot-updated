import logging
import re
from typing import List, Optional

import httpx
from cachetools import LRUCache

from app.config import get_settings
from models.schemas import Citation, QueryResponse
from rag.extractive import build_extractive_answer
from rag.llm import MedicalLLM
from rag.scoring import compute_retrieval_confidence, retrieval_is_sufficient
from rag.vector_store import FAISSVectorStore, RetrievedChunk
from services.embeddings import EmbeddingServiceError, OpenAIQuotaError
from rag.formatting import paragraph_to_bullets
from services.guardrails import validate_answer_grounding, is_greeting_or_meta, check_query_safety, QueryAnalysisResult
from services.pubmed import prioritize_trusted_journals, search_pubmed
from rag.response_generator import generate_response
from rag.query_quality import assess_query_quality
from services.reranker import get_reranker

logger = logging.getLogger(__name__)

# Cache the last 5 FAISS indices to prevent massive memory spikes and CPU usage on repeated queries
_vector_store_cache = LRUCache(maxsize=5)

class RAGPipeline:
    def __init__(self) -> None:
        self._llm: Optional[MedicalLLM] = None

    def _get_llm(self) -> MedicalLLM:
        if self._llm is None:
            self._llm = MedicalLLM()
        return self._llm

    async def run(self, query: str, max_papers: Optional[int] = None, risk_level: str = "LOW", analysis: Optional[QueryAnalysisResult] = None) -> QueryResponse:
        settings = get_settings()

        # Clean query by stripping outer quotes and whitespace
        query = query.strip().strip('"').strip("'").strip()

        is_valid, reason, q_score, ignored_tokens = assess_query_quality(query)
        logger.info("Query quality score: %.2f (Ignored tokens: %s)", q_score, ignored_tokens)
        logger.info("Assessed risk level: %s", risk_level)
        
        if not is_valid and not is_greeting_or_meta(query):
            return generate_response(
                answer=reason,
                citations=[],
                confidence_score=0.0,
                insufficient_evidence=False,
                confidence_note="Clarification required.",
                sources_searched=[],
            )
            
        # Clean query
        for token in ignored_tokens:
            query = re.sub(r'\b' + re.escape(token) + r'\b', '', query, flags=re.IGNORECASE).strip()
        query = re.sub(r'\s+', ' ', query).strip()

        # Handle greetings/meta queries early to skip PubMed search
        if is_greeting_or_meta(query):
            answer, cited_indices, _ = await self._get_llm().generate(query, [])
            return generate_response(
                answer=answer,
                citations=[],
                confidence_score=1.0,
                insufficient_evidence=False,
                confidence_note="Assistant information.",
                sources_searched=[],
            )

        # 1. Preprocess: Get or generate query analysis dynamically
        if analysis is None:
            analysis = await check_query_safety(query, block_emergency=False)
            
        translated_query = analysis.simplified_search_query or query
        inferred_diseases = analysis.inferred_diseases
        
        # Build Boolean search query and normalized query dynamically
        expanded_query = self._build_dynamic_boolean_query(translated_query, analysis.synonym_expansion, inferred_diseases)
        normalized_query = translated_query
        
        logger.info("Raw query: %s", query)
        logger.info("Dynamic translated query: %s", translated_query)
        logger.info("Normalized query: %s", normalized_query)
        if inferred_diseases:
            logger.info("Dynamic inferred diseases: %s", ", ".join(inferred_diseases))
        logger.info("Dynamic expanded PubMed query: %s", expanded_query)

        # 2. Search PubMed using the high-quality Boolean expanded query
        papers = await search_pubmed(expanded_query, max_results=max_papers, raw_query=query)
        papers = prioritize_trusted_journals(papers)

        # 3. Handle zero search results fallback
        if not papers:
            return generate_response(
                answer="Current evidence is insufficient to provide a reliable answer.",
                citations=[],
                confidence_note=(
                    "No relevant papers were retrieved from PubMed for this query. "
                    "Try broader or alternative medical terminology."
                ),
                confidence_score=0.0,
                insufficient_evidence=True,
                sources_searched=["PubMed"],
            )

        # 4. Semantic hybrid search and Reranking
        try:
            vector_store = _vector_store_cache.get(expanded_query)
            if vector_store is None:
                vector_store = FAISSVectorStore()
                await vector_store.build_from_papers(papers)
                _vector_store_cache[expanded_query] = vector_store
            else:
                logger.info("Using cached FAISS index for query: %s", expanded_query)

            # Fetch top candidates from hybrid search (FAISS Cosine + BM25 RRF)
            hybrid_chunks = await vector_store.search(
                normalized_query, top_k=20
            )
            
            # Apply Cross-Encoder Reranker
            if hybrid_chunks:
                logger.info("Reranking %d candidate chunks", len(hybrid_chunks))
                texts = [c.text for c in hybrid_chunks]
                reranker = get_reranker()
                rerank_scores = reranker.safe_rerank(normalized_query, texts)
                
                if rerank_scores:
                    # Overwrite RRF/FAISS scores with high-quality cross-encoder sigmoid scores
                    for chunk, r_score in zip(hybrid_chunks, rerank_scores):
                        chunk.score = float(r_score)
                    hybrid_chunks.sort(key=lambda c: c.score, reverse=True)
                else:
                    logger.warning("Skipped reranking. Falling back to FAISS/BM25 scores.")

            # Filter candidates through strict medical relevance checker
            if hybrid_chunks:
                # 1. Identify the top 5 unique papers to check in a batch
                unique_papers = []
                seen_pmids = set()
                for chunk in hybrid_chunks:
                    p = chunk.paper
                    if p.pmid not in seen_pmids:
                        seen_pmids.add(p.pmid)
                        unique_papers.append(p)
                        if len(unique_papers) >= 5:
                            break

                # 2. Call batch relevance check in a single LLM request
                from services.relevance import check_papers_relevance_batch
                batch_results = await check_papers_relevance_batch(normalized_query, unique_papers)
                
                # 3. Build relevance map
                relevance_map = {}  # pmid -> bool
                for pmid, (is_rel, reason) in batch_results.items():
                    relevance_map[pmid] = is_rel
                    if not is_rel:
                        logger.info("Paper PMID %s rejected: %s", pmid, reason)

                # Filter both the retrieved chunks and the original papers list
                filtered_hybrid_chunks = []
                for chunk in hybrid_chunks:
                    pmid = chunk.paper.pmid
                    if relevance_map.get(pmid, False):
                        filtered_hybrid_chunks.append(chunk)
                
                hybrid_chunks = filtered_hybrid_chunks
                papers = [p for p in papers if relevance_map.get(p.pmid, False)]

            chunks = hybrid_chunks[:settings.pubmed_retrieval_top_k]
            
        except (OpenAIQuotaError, EmbeddingServiceError) as exc:
            logger.error("Embedding failed: %s", exc)
            raise

        # Always use the best-matching chunks (do not over-filter TF-IDF/dense scores)
        evidence_chunks = chunks[: max(settings.min_evidence_chunks, 5)]
        has_disease = len(inferred_diseases) > 0
        confidence_score = compute_retrieval_confidence(
            chunks, papers, use_local=False,
            query_quality_score=q_score,
            risk_level=risk_level,
            has_inferred_disease=has_disease
        )

        for chunk in chunks:
            logger.debug("Chunk %d (PMID: %s) score: %.3f", chunk.chunk_index, chunk.paper.pmid, chunk.score)
            
        evidence_pmids = set(c.paper.pmid for c in evidence_chunks)
        discarded_papers = set(p.pmid for p in papers) - evidence_pmids
        if discarded_papers:
            logger.debug("Discarded %d citations with low semantic relevance: %s", len(discarded_papers), discarded_papers)

        # 5. Handle low-confidence / insufficient semantic match fallback
        if not retrieval_is_sufficient(chunks, papers, use_local=False):
            logger.info(
                "Weak retrieval: top_score=%.3f papers=%d",
                max((c.score for c in chunks), default=0),
                len(papers),
            )
            return generate_response(
                answer="Current evidence is insufficient to provide a reliable answer.",
                citations=self._chunks_to_citations(evidence_chunks[:3]),
                confidence_note=(
                    "Retrieved literature had very low semantic match to your question. "
                    "Try rephrasing with medical subject terms (drug class, condition, outcome)."
                ),
                confidence_score=confidence_score,
                insufficient_evidence=True,
                sources_searched=["PubMed"],
            )

        # 6. Generate the grounded answer from candidate chunks
        try:
            answer, cited_indices, llm_insufficient = await self._get_llm().generate(
                normalized_query, evidence_chunks
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("LLM quota or rate limit exceeded; using extractive fallback")
                answer, cited_indices = build_extractive_answer(normalized_query, evidence_chunks)
                llm_insufficient = False
            else:
                raise

        cited_chunks = self._resolve_citations(evidence_chunks, cited_indices)
        if not cited_chunks:
            cited_chunks = evidence_chunks[:3]

        citations = self._chunks_to_citations(cited_chunks)

        if not llm_insufficient and "\n- " not in answer and not answer.startswith("- "):
            answer = paragraph_to_bullets(answer)

        final_answer, insufficient, confidence_note = validate_answer_grounding(
            answer,
            llm_insufficient,
            confidence_score,
        )

        if risk_level == "HIGH":
            confidence_note = "High-risk query detected. Confidence strictly reduced. " + confidence_note

        # 7. Package and label the response via response_generator
        response = generate_response(
            answer=final_answer,
            citations=citations,
            confidence_note=confidence_note,
            confidence_score=confidence_score,
            insufficient_evidence=insufficient,
            sources_searched=["PubMed (BMJ, Nature, Lancet, and peer-reviewed literature)"],
        )
        
        if risk_level == "HIGH":
            response.answer += "\n\n**Important Safety Notice:** Current evidence does not support replacing evidence-based treatment with unverified alternatives. Supportive approaches may help symptom management, but treatment decisions should be made with a licensed clinician or specialist."
            
        if risk_level == "PATIENT_SPECIFIC":
            response.answer += "\n\n**Disclaimer:** The output is for information purposes only and does not constitute medical advice, diagnosis, or treatment. Always verify information against primary, peer-reviewed clinical guidelines."
            
        return response

    def _resolve_citations(
        self, chunks: List[RetrievedChunk], indices: List[int]
    ) -> List[RetrievedChunk]:
        result: List[RetrievedChunk] = []
        seen = set()
        for idx in indices:
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1]
                if chunk.paper.pmid not in seen:
                    seen.add(chunk.paper.pmid)
                    result.append(chunk)
        return result

    @staticmethod
    def _chunks_to_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for chunk in chunks:
            p = chunk.paper
            if p.pmid in seen:
                continue
            seen.add(p.pmid)
            citations.append(
                Citation(
                    title=p.title,
                    journal=p.journal,
                    year=p.year,
                    pubmed_url=p.pubmed_url,
                    pmid=p.pmid,
                    authors=p.authors,
                )
            )
        return citations

    @staticmethod
    def _build_dynamic_boolean_query(
        simplified_query: str,
        synonym_expansion: dict,
        inferred_diseases: List[str]
    ) -> str:
        # Clean up input keywords
        keywords = [kw.strip() for kw in simplified_query.split() if kw.strip()]
        if not keywords:
            return simplified_query

        concept_groups = []
        for kw in keywords:
            # Check for synonyms dynamically from the LLM dictionary
            syns = synonym_expansion.get(kw, [])
            # Make sure keys are matched case-insensitively
            if not syns:
                for key, val in synonym_expansion.items():
                    if key.lower() == kw.lower():
                        syns = val
                        break
            
            # Ensure the keyword itself is in the synonyms list
            if kw not in syns:
                # Avoid inserting a duplicate that is case-insensitive-only
                if not any(s.lower() == kw.lower() for s in syns):
                    syns = [kw] + syns
            
            # format synonyms: wrap in double quotes if it contains spaces
            formatted = [f'"{s}"' if " " in s else s for s in syns]
            concept_groups.append("(" + " OR ".join(formatted) + ")")

        disease_terms = []
        for disease in inferred_diseases:
            # Check if this disease is already represented in search keywords to avoid redundant groups
            disease_lower = disease.lower()
            if any(disease_lower == kw.lower() for kw in keywords):
                continue
                
            syns = synonym_expansion.get(disease, [])
            if not syns:
                for key, val in synonym_expansion.items():
                    if key.lower() == disease_lower:
                        syns = val
                        break
            if disease not in syns:
                if not any(s.lower() == disease_lower for s in syns):
                    syns = [disease] + syns
            formatted = [f'"{s}"' if " " in s else s for s in syns]
            disease_terms.extend(formatted)

        if disease_terms:
            concept_groups.append("(" + " OR ".join(disease_terms) + ")")

        if not concept_groups:
            return simplified_query

        return " AND ".join(concept_groups)
