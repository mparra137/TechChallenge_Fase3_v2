"""Criação lazy e reutilizável de embeddings Hugging Face."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src import config
from src.utils.logger import setup_logger

logger = setup_logger("rag.embeddings")


class EmbeddingService:
    """Own a single HuggingFaceEmbeddings instance for the current process."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._embeddings: Any | None = None

    def get_embeddings(self) -> Any:
        """Initialize the configured embedding model only when it is first needed."""
        if self._embeddings is not None:
            return self._embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Não foi possível inicializar embeddings '{self.model_name}' no dispositivo '{self.device}'."
            ) from exc
        logger.info("Modelo de embeddings carregado: %s (%s)", self.model_name, self.device)
        return self._embeddings


@lru_cache(maxsize=8)
def get_embedding_service(
    model_name: str = config.EMBEDDING_MODEL,
    device: str = config.EMBEDDING_DEVICE,
) -> EmbeddingService:
    """Return a cached service so repeated queries do not reload the embedding model."""
    return EmbeddingService(model_name=model_name, device=device)
