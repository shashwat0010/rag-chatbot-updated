import asyncio
import contextvars
from contextlib import asynccontextmanager
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# Global ContextVar for tracing request IDs
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="system"
)

# Global Semaphore for Mistral API
_mistral_semaphore = None


def _get_semaphore() -> asyncio.Semaphore:
    global _mistral_semaphore
    if _mistral_semaphore is None:
        settings = get_settings()
        limit = settings.mistral_concurrency_limit
        _mistral_semaphore = asyncio.Semaphore(limit)
        logger.info("Initialized global Mistral semaphore with limit: %d", limit)
    return _mistral_semaphore


@asynccontextmanager
async def get_mistral_semaphore():
    sem = _get_semaphore()
    logger.info("waiting for semaphore")
    async with sem:
        logger.info("semaphore acquired")
        try:
            yield
        finally:
            logger.info("semaphore released")
