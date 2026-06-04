import logging
import math
import psutil
from typing import List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Global singleton instance
_reranker_instance = None


def get_reranker() -> "RerankerService":
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance


class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def load_model(self):
        if self._model is None:
            mem_before = psutil.virtual_memory().used / (1024 * 1024)
            logger.info("Loading reranker model: %s (RAM before: %.1f MB)", self.model_name, mem_before)
            self._model = CrossEncoder(self.model_name, max_length=512)
            mem_after = psutil.virtual_memory().used / (1024 * 1024)
            logger.info("Reranker model loaded. (RAM after: %.1f MB, Delta: %.1f MB)", mem_after, mem_after - mem_before)
        return self._model

    def safe_rerank(self, query: str, texts: List[str], max_ram_percent: float = 95.0) -> List[float]:
        if not texts:
            return []

        mem_percent = psutil.virtual_memory().percent
        if mem_percent > max_ram_percent:
            logger.warning("RAM usage at %.1f%% (> %.1f%%). Skipping reranker to prevent OOM.", mem_percent, max_ram_percent)
            return []

        try:
            model = self.load_model()
            pairs = [[query, text] for text in texts]
            scores = model.predict(pairs)
            return [1.0 / (1.0 + math.exp(-s)) for s in scores]
        except Exception as exc:
            logger.error("Error during reranking: %s", exc)
            return []
