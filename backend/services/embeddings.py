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
    Call Mistral AI Embeddings API (free tier).
    Model: mistral-embed (1024-dim, high quality, OpenAI-compatible endpoint).
    Get a free key at: https://console.mistral.ai/
    """
    settings = get_settings()
    if not settings.mistral_api_key or settings.mistral_api_key == "your_mistral_key_here":
        raise EmbeddingServiceError(
            "Mistral API key is not configured. "
            "Get a free key at https://console.mistral.ai/ "
            "and set MISTRAL_API_KEY in backend/.env"
        )

    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_EMBED_MODEL,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        try:
            response = await _do_embed_request(client, MISTRAL_EMBED_URL, headers, payload)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise EmbeddingServiceError(
                    "Invalid Mistral API key. Check MISTRAL_API_KEY in backend/.env"
                ) from exc
            if status == 429:
                raise EmbeddingServiceError(
                    "Mistral API rate limit hit. Please wait a moment and retry."
                ) from exc
            raise EmbeddingServiceError(
                f"Mistral Embeddings API error ({status}): {exc.response.text[:200]}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError(
                "Mistral Embeddings API timed out. Try again."
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingServiceError(
                f"Cannot reach Mistral Embeddings API: {exc}"
            ) from exc

    data = response.json()
    # Mistral returns {"data": [{"embedding": [...], "index": 0}, ...]}
    embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    logger.info("Got %d embeddings from Mistral (dim=%d)", len(embeddings), len(embeddings[0]) if embeddings else 0)
    return embeddings


async def embed_texts(texts: List[str], *, allow_local_fallback: bool = True) -> List[List[float]]:
    """Embed a batch of document texts using Mistral AI Embeddings API (free)."""
    if not texts:
        return []
    logger.info("Embedding %d texts via Mistral AI", len(texts))
    return await _mistral_embed(texts)


async def embed_query(text: str) -> List[List[float]]:
    """Embed a single query string using Mistral AI Embeddings API."""
    return await _mistral_embed([text])
