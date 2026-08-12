import logging
from typing import List, Optional
import numpy as np

from app.config import get_settings
from models.schemas import PaperMetadata
from services.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    logger.warning("Pinecone client not installed. Falling back to local vector store.")


class PineconeVectorStore:
    def __init__(self) -> None:
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name
        self._index = None
        self._is_initialized = False

        if HAS_PINECONE and self.api_key:
            try:
                self.pc = Pinecone(api_key=self.api_key)
                existing_indices = [idx.name for idx in self.pc.list_indexes()]
                if self.index_name in existing_indices:
                    self._index = self.pc.Index(self.index_name)
                    self._is_initialized = True
                    logger.info("Pinecone index '%s' connected successfully.", self.index_name)
                else:
                    logger.info("Pinecone ready. Index '%s' will be created on first upsert.", self.index_name)
                    self._is_initialized = True
            except Exception as e:
                logger.warning("Failed to initialize Pinecone vector store: %s", e)
                self._is_initialized = False

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self._index is not None

    async def upsert_chunks(self, chunks: List[dict]) -> None:
        """
        Upserts chunk vectors and metadata into Pinecone.
        Each chunk dict should contain: id, vector, metadata (pmid, title, journal, year, text, chunk_index)
        """
        if not self.is_ready or not chunks:
            return

        try:
            if self._index is None and self.pc:
                dimension = len(chunks[0]["vector"])
                existing_indices = [idx.name for idx in self.pc.list_indexes()]
                if self.index_name not in existing_indices:
                    logger.info("Creating Pinecone index '%s' with dim=%d...", self.index_name, dimension)
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                self._index = self.pc.Index(self.index_name)

            vectors_to_upsert = []
            for item in chunks:
                vectors_to_upsert.append((
                    item["id"],
                    item["vector"],
                    item["metadata"]
                ))
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self._index.upsert(vectors=batch)
            logger.info("Successfully upserted %d vectors to Pinecone.", len(vectors_to_upsert))
        except Exception as e:
            logger.error("Error upserting vectors to Pinecone: %s", e)

    async def search(self, query_vector: List[float], top_k: int = 8) -> List[dict]:
        """
        Queries Pinecone for nearest vector matches and returns metadata hits.
        """
        if not self.is_ready:
            return []

        try:
            res = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            matches = []
            for match in res.get("matches", []):
                matches.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"]
                })
            return matches
        except Exception as e:
            logger.error("Error searching Pinecone vector store: %s", e)
            return []
