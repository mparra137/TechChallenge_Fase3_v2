"""Unit tests for cached embedding service construction without model downloads."""

from unittest.mock import MagicMock, patch
import os
from math import hypot
import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.embeddings import get_embedding_service


def test_embedding_instance_is_created_once_per_service() -> None:
    """The expensive HuggingFace model must not be rebuilt for repeated calls."""
    fake_embeddings = MagicMock()
    with patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=fake_embeddings) as mocked:
        service = EmbeddingService("test-model", "cpu")

        assert service.get_embeddings() is fake_embeddings
        assert service.get_embeddings() is fake_embeddings

    mocked.assert_called_once_with(
        model_name="test-model",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.mark.skipif(os.getenv("TEST_REAL_MINILM") != "1", reason="Integração opt-in com o MiniLM em cache")
def test_real_model_dimension_and_norm():
    vector = get_embedding_service().get_embeddings().embed_query("registro hospitalar sintético")
    assert len(vector) == 384
    assert hypot(*vector) == pytest.approx(1.0, abs=1e-5)
