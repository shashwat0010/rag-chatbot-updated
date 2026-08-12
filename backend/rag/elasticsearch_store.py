import logging
from typing import List, Optional, Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from elasticsearch import AsyncElasticsearch, helpers
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False
    logger.warning("Elasticsearch client not installed. Falling back to local BM25 store.")


class ElasticsearchStore:
    def __init__(self) -> None:
        self.es_url = settings.elasticsearch_url
        self.index_name = settings.elasticsearch_index
        self.es: Optional[Any] = None
        self._is_initialized = False

        if HAS_ELASTICSEARCH and self.es_url:
            try:
                kwargs = {"request_timeout": 5.0, "max_retries": 1}
                if settings.elasticsearch_api_key:
                    kwargs["api_key"] = settings.elasticsearch_api_key
                self.es = AsyncElasticsearch(self.es_url, **kwargs)
                self._is_initialized = True
            except Exception as e:
                logger.warning("Elasticsearch connection initialization error: %s", e)

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self.es is not None

    async def init_index(self) -> None:
        if not self.is_ready:
            return
        try:
            exists = await self.es.indices.exists(index=self.index_name)
            if not exists:
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "pmid": {"type": "keyword"},
                            "title": {"type": "text"},
                            "journal": {"type": "keyword"},
                            "year": {"type": "integer"},
                            "abstract": {"type": "text"},
                            "chunk_index": {"type": "integer"},
                            "vector": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
                await self.es.indices.create(index=self.index_name, body=mapping)
                logger.info("Created Elasticsearch index: %s", self.index_name)
        except Exception as e:
            logger.warning("Error creating Elasticsearch index mapping: %s", e)

    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not self.is_ready:
            return
        try:
            await self.init_index()
            actions = []
            for item in chunks:
                meta = item["metadata"]
                action = {
                    "_index": self.index_name,
                    "_id": item["id"],
                    "_source": {
                        "pmid": meta.get("pmid"),
                        "title": meta.get("title"),
                        "journal": meta.get("journal"),
                        "year": int(meta["year"]) if meta.get("year") and str(meta["year"]).isdigit() else None,
                        "abstract": meta.get("abstract"),
                        "text": meta.get("text"),
                        "chunk_index": meta.get("chunk_index"),
                        "vector": item.get("vector")
                    }
                }
                actions.append(action)

            if actions:
                await helpers.async_bulk(self.es, actions)
                await self.es.indices.refresh(index=self.index_name)
                logger.info("Indexed %d chunks into Elasticsearch", len(actions))
        except Exception as e:
            logger.error("Failed indexing into Elasticsearch: %s", e)

    async def search_bm25(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Fast BM25 full-text keyword retrieval in Elasticsearch.
        """
        if not self.is_ready:
            return []
        try:
            body = {
                "query": {
                    "match": {
                        "text": {
                            "query": query
                        }
                    }
                },
                "size": top_k
            }
            res = await self.es.search(index=self.index_name, body=body)
            hits = []
            for hit in res["hits"]["hits"]:
                hits.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                })
            return hits
        except Exception as e:
            logger.error("Elasticsearch BM25 query error: %s", e)
            return []

    async def close(self) -> None:
        if self.es:
            await self.es.close()
