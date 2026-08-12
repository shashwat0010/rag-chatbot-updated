import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np
import psutil
from rank_bm25 import BM25Okapi

from models.schemas import PaperMetadata
from services.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    paper: PaperMetadata
    text: str
    score: float
    chunk_index: int


def _paper_to_chunks(paper: PaperMetadata) -> List[str]:
    header = f"Title: {paper.title}\nJournal: {paper.journal}\nYear: {paper.year or 'N/A'}\n"
    body = paper.abstract or ""
    full_text = header + body
    
    max_len = 800
    overlap = 100
    
    if len(full_text) <= max_len:
        return [full_text]
        
    chunks: List[str] = []
    
    # Split the text on sentence boundaries: periods, question marks, or exclamation marks followed by spaces/newlines
    sentence_endings = re.compile(r'(?<=[.?!])\s+')
    sentences = sentence_endings.split(full_text)
    
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds max_len
        if current_len + sentence_len + 1 > max_len:
            if current_chunk:
                # Flush the current chunk
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                
                # Slide window overlap: keep sentences at the end of the previous chunk that fit within the overlap size
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) + 1 <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len
            
            # If a single sentence exceeds max_len, fall back to character split for that sentence
            if sentence_len > max_len:
                start = 0
                while start < sentence_len:
                    end = min(start + max_len, sentence_len)
                    chunks.append(sentence[start:end])
                    if end >= sentence_len:
                        break
                    start = end - overlap
                current_chunk = []
                current_len = 0
            else:
                current_chunk.append(sentence)
                current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


import uuid
from rag.pinecone_store import PineconeVectorStore
from rag.elasticsearch_store import ElasticsearchStore


class FAISSVectorStore:
    def __init__(self) -> None:
        self._index: Optional[faiss.IndexFlatIP] = None
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[RetrievedChunk] = []
        self._dimension: int = 0
        self.pinecone_store = PineconeVectorStore()
        self.elasticsearch_store = ElasticsearchStore()

    @property
    def is_ready(self) -> bool:
        if self.pinecone_store.is_ready or self.elasticsearch_store.is_ready:
            return True
        return self._index is not None and len(self._chunks) > 0

    async def build_from_papers(self, papers: List[PaperMetadata]) -> None:
        self._chunks = []
        texts: List[str] = []
        chunk_objects = []

        for paper in papers:
            for i, chunk_text in enumerate(_paper_to_chunks(paper)):
                retrieved_chunk = RetrievedChunk(paper=paper, text=chunk_text, score=0.0, chunk_index=i)
                self._chunks.append(retrieved_chunk)
                texts.append(chunk_text)
                chunk_objects.append((paper, chunk_text, i))

        if not texts:
            self._index = None
            self._bm25 = None
            return

        mem_before = psutil.virtual_memory().used / (1024 * 1024)
        logger.info("Building vector and keyword search indices. RAM before: %.1f MB", mem_before)

        # Generate dense embeddings
        embeddings = await embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)

        # 1. Index to Pinecone & Elasticsearch if available
        pinecone_payloads = []
        for idx, ((paper, chunk_text, chunk_idx), vec) in enumerate(zip(chunk_objects, embeddings)):
            chunk_id = f"chunk_{paper.pmid}_{chunk_idx}_{uuid.uuid4().hex[:6]}"
            pinecone_payloads.append({
                "id": chunk_id,
                "vector": vec,
                "metadata": {
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "journal": paper.journal,
                    "year": str(paper.year or ""),
                    "abstract": paper.abstract or "",
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                }
            })

        if self.pinecone_store.is_ready:
            await self.pinecone_store.upsert_chunks(pinecone_payloads)

        if self.elasticsearch_store.is_ready:
            await self.elasticsearch_store.index_chunks(pinecone_payloads)

        # 2. Local Fallback Index (FAISS + rank-bm25)
        tokenized_texts = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized_texts)

        faiss.normalize_L2(vectors)
        self._dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        mem_after = psutil.virtual_memory().used / (1024 * 1024)
        logger.info("Built vector indices with %d chunks. RAM after: %.1f MB", len(texts), mem_after)

    async def search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        # If Pinecone or Elasticsearch are available, use persistent hybrid search
        if self.pinecone_store.is_ready or self.elasticsearch_store.is_ready:
            return await self._persistent_hybrid_search(query, top_k)

        # Otherwise fallback to local FAISS + BM25Okapi
        if not self.is_ready or self._index is None or self._bm25 is None:
            return []

        fusion_k = min(30, len(self._chunks))

        # 1. FAISS Dense Search
        query_vecs = await embed_query(query)
        q = np.array(query_vecs, dtype=np.float32)
        faiss.normalize_L2(q)
        
        faiss_scores, faiss_indices = self._index.search(q, fusion_k)
        faiss_ranks = {}
        for rank, idx in enumerate(faiss_indices[0]):
            if 0 <= idx < len(self._chunks):
                faiss_ranks[idx] = rank + 1

        # 2. BM25 Sparse Search
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_ranked_indices[:fusion_k]):
            bm25_ranks[idx] = rank + 1

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_k = 60
        for idx in range(len(self._chunks)):
            if idx in faiss_ranks or idx in bm25_ranks:
                f_rank = faiss_ranks.get(idx, 1000)
                b_rank = bm25_ranks.get(idx, 1000)
                if idx in bm25_ranks and bm25_scores[idx] <= 0:
                    b_rank = 1000
                rrf_scores[idx] = (1.0 / (rrf_k + f_rank)) + (1.0 / (rrf_k + b_rank))

        fused_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        results: List[RetrievedChunk] = []
        seen_pmids = set()
        for idx in fused_indices:
            chunk = self._chunks[idx]
            if chunk.paper.pmid in seen_pmids and len(results) >= top_k:
                continue
            seen_pmids.add(chunk.paper.pmid)
            orig_score = float(faiss_scores[0][list(faiss_indices[0]).index(idx)]) if idx in faiss_indices[0] else 0.5
            
            results.append(
                RetrievedChunk(
                    paper=chunk.paper,
                    text=chunk.text,
                    score=orig_score,
                    chunk_index=chunk.chunk_index,
                )
            )
            if len(results) >= top_k:
                break

        return results

    async def _persistent_hybrid_search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        query_vec = await embed_query(query)
        
        pinecone_hits = []
        if self.pinecone_store.is_ready:
            pinecone_hits = await self.pinecone_store.search(query_vec[0], top_k=top_k * 2)

        es_hits = []
        if self.elasticsearch_store.is_ready:
            es_hits = await self.elasticsearch_store.search_bm25(query, top_k=top_k * 2)

        # If both returns hits, fuse via RRF
        combined_docs = {}
        for rank, hit in enumerate(pinecone_hits):
            meta = hit["metadata"]
            doc_key = f"{meta.get('pmid')}_{meta.get('chunk_index')}"
            combined_docs[doc_key] = {
                "meta": meta,
                "score": 1.0 / (60 + rank + 1)
            }

        for rank, hit in enumerate(es_hits):
            src = hit["source"]
            doc_key = f"{src.get('pmid')}_{src.get('chunk_index')}"
            score = 1.0 / (60 + rank + 1)
            if doc_key in combined_docs:
                combined_docs[doc_key]["score"] += score
            else:
                combined_docs[doc_key] = {
                    "meta": src,
                    "score": score
                }

        # Sort by fused score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x["score"], reverse=True)

        results: List[RetrievedChunk] = []
        seen_pmids = set()
        for item in sorted_docs:
            m = item["meta"]
            pmid = m.get("pmid", "")
            if pmid in seen_pmids and len(results) >= top_k:
                continue
            seen_pmids.add(pmid)

            paper = PaperMetadata(
                pmid=pmid,
                title=m.get("title", ""),
                journal=m.get("journal", ""),
                year=int(m["year"]) if m.get("year") and str(m["year"]).isdigit() else None,
                abstract=m.get("abstract", ""),
                pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )

            results.append(
                RetrievedChunk(
                    paper=paper,
                    text=m.get("text", ""),
                    score=item["score"],
                    chunk_index=int(m.get("chunk_index", 0)),
                )
            )
            if len(results) >= top_k:
                break

        return results

