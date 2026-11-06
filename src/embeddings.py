"""Create embeddings with the Gemini API."""
import time
from functools import lru_cache

import httpx
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src import config

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "QUESTION_ANSWERING"
BATCH_SIZE = 20
MAX_RETRIES = 6
RETRY_CODES = {429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    """Embedding failed. The message is safe to show to users."""


@lru_cache(maxsize=1)
def _client():
    if not config.GEMINI_API_KEY:
        raise EmbeddingError(
            "The Gemini API key is missing. Add GEMINI_API_KEY to the .env file."
        )
    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=60_000),  # 60 seconds
    )


def _is_retryable(exc) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in RETRY_CODES
    return isinstance(exc, httpx.TransportError)  # timeouts, dropped connections


def _friendly(exc) -> str:
    code = getattr(exc, "code", None)
    if code in (401, 403):
        return "The Gemini API key was rejected. Check GEMINI_API_KEY in the .env file."
    if code == 404:
        return f"Embedding model '{config.EMBED_MODEL}' was not found. Check EMBED_MODEL in .env."
    if code == 429:
        return "The Gemini rate limit was reached. Wait a minute and try again."
    return "The embedding service is not available right now. Please try again later."


def _normalize(vectors):
    """gemini-embedding-001 needs manual normalisation below 3072 dimensions."""
    array = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).tolist()


def _embed_batch(texts, task_type):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client().models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBED_DIMENSIONS,
                ),
            )
            vectors = [e.values for e in response.embeddings]
            if len(vectors) != len(texts):
                raise EmbeddingError("The embedding service returned an unexpected result.")
            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                raise EmbeddingError(_friendly(exc)) from exc
            wait = min(60, 10 * attempt) if getattr(exc, "code", None) == 429 else 5 * attempt
            print(f"    Embedding API busy (code {getattr(exc, 'code', None)}); retry {attempt} in {wait}s")
            time.sleep(wait)


def embed_documents(texts, show_progress: bool = True):
    """Embed many chunks. Returns one normalised vector per text."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors.extend(_normalize(_embed_batch(batch, DOCUMENT_TASK_TYPE)))
        if show_progress:
            print(f"    embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
        time.sleep(0.5)
    return vectors


def embed_query(text: str):
    """Embed one student question."""
    return _normalize(_embed_batch([text], QUERY_TASK_TYPE))[0]