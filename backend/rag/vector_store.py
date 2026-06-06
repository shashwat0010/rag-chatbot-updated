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


class FAISSVectorStore:
    def __init__(self) -> None:
        self._index: Optional[faiss.IndexFlatIP] = None
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[RetrievedChunk] = []
        self._dimension: int = 0

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    async def build_from_papers(self, papers: List[PaperMetadata]) -> None:
        self._chunks = []
        texts: List[str] = []
        for paper in papers:
            for i, chunk_text in enumerate(_paper_to_chunks(paper)):
                self._chunks.append(
                    RetrievedChunk(paper=paper, text=chunk_text, score=0.0, chunk_index=i)
                )
                texts.append(chunk_text)

        if not texts:
            self._index = None
            self._bm25 = None
            return

        mem_before = psutil.virtual_memory().used / (1024 * 1024)
        logger.info("Building BM25 and FAISS indices. RAM before: %.1f MB", mem_before)

        # 1. Build BM25 Index
        tokenized_texts = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized_texts)

        # 2. Build FAISS Index

        embeddings = await embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        mem_after = psutil.virtual_memory().used / (1024 * 1024)
        logger.info("Built FAISS index with %d chunks. RAM after: %.1f MB (Delta: %.1f MB)", len(texts), mem_after, mem_after - mem_before)

    async def search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
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
                # Ensure BM25 score > 0 to be considered a hit, otherwise heavily penalize
                if idx in bm25_ranks and bm25_scores[idx] <= 0:
                    b_rank = 1000
                rrf_scores[idx] = (1.0 / (rrf_k + f_rank)) + (1.0 / (rrf_k + b_rank))

        # Sort by fused score
        fused_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        # 4. Build Results
        results: List[RetrievedChunk] = []
        seen_pmids = set()
        for idx in fused_indices:
            chunk = self._chunks[idx]
            if chunk.paper.pmid in seen_pmids and len(results) >= top_k:
                continue
            seen_pmids.add(chunk.paper.pmid)
            # Use original FAISS score for fallback compatibility, reranker will override it anyway
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
