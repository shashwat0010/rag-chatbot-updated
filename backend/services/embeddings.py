import logging
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.config import get_settings
from app.concurrency import get_mistral_semaphore

logger = logging.getLogger(__name__)

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_EMBED_MODEL = "mistral-embed"
HF_TIMEOUT = 30.0


class EmbeddingServiceError(Exception):
    """Raised when embeddings cannot be computed."""


class OpenAIQuotaError(EmbeddingServiceError):
    """Kept for backward compatibility with pipeline error handling."""


def _is_retryable_httpx_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    retry=retry_if_exception(_is_retryable_httpx_error),
    reraise=True
)
async def _do_embed_request(client: httpx.AsyncClient, url: str, headers: dict, payload: dict):
    async with get_mistral_semaphore():
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response


async def _mistral_embed(texts: List[str]) -> List[List[float]]:
    """
    Call Mistral AI or OpenRouter Embeddings API, falling back to TF-IDF embeddings on any error.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key or settings.mistral_api_key
    if not api_key or api_key == "your_mistral_key_here":
        from services.local_embeddings import get_local_embeddings
        return get_local_embeddings(texts)

    if api_key.startswith("sk-or-") or settings.openrouter_api_key:
        embed_url = f"{settings.openrouter_base_url.rstrip('/')}/embeddings"
        model_name = "openai/text-embedding-3-small"
    else:
        embed_url = MISTRAL_EMBED_URL
        model_name = MISTRAL_EMBED_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "input": texts,
    }

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            response = await _do_embed_request(client, embed_url, headers, payload)
            data = response.json()
            if "data" in data and data["data"]:
                embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                logger.info("Got %d embeddings from API (dim=%d)", len(embeddings), len(embeddings[0]) if embeddings else 0)
                return embeddings
    except Exception as exc:
        logger.warning("Embeddings API call failed (%s). Using local TF-IDF fallback.", exc)

    from services.local_embeddings import get_local_embeddings
    return get_local_embeddings(texts)


async def embed_texts(texts: List[str], *, allow_local_fallback: bool = True) -> List[List[float]]:
    """Embed a batch of document texts using Mistral AI / OpenRouter Embeddings API, with local fallback."""
    if not texts:
        return []
    settings = get_settings()
    api_key = settings.openrouter_api_key or settings.mistral_api_key
    if not api_key or api_key == "your_mistral_key_here":
        logger.info("No API key configured for embeddings. Using local TF-IDF embedding fallback.")
        from services.local_embeddings import get_local_embeddings
        return get_local_embeddings(texts)

    try:
        logger.info("Embedding %d texts via Mistral AI", len(texts))
        return await _mistral_embed(texts)
    except Exception as e:
        if allow_local_fallback:
            logger.warning("Embedding API request failed (%s). Falling back to local embeddings.", e)
            from services.local_embeddings import get_local_embeddings
            return get_local_embeddings(texts)
        raise


async def embed_query(text: str) -> List[List[float]]:
    """Embed a single query string using Embeddings API with fallback."""
    return await embed_texts([text], allow_local_fallback=True)

