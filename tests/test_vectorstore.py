"""Chroma tests using deterministic local embeddings and no network access."""

from pathlib import Path
import tempfile

from langchain_core.documents import Document

from src.vectorstore.chroma_service import ChromaService


class FakeEmbeddings:
    """Small normalized embedding implementation with the expected 384 dimensions."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = sum(ord(character) for character in text) % 384
        vector = [0.0] * 384
        vector[seed] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeEmbeddingService:
    """Inject a local deterministic embedding function into the real Chroma service."""

    model_name = "test-384"

    def get_embeddings(self) -> FakeEmbeddings:
        return FakeEmbeddings()


def _document(document_id: str, patient_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "patient_id": patient_id,
            "patient_name": "Paciente Teste",
            "document_type": "exame_laboratorial",
            "document_date": "2026-01-01",
            "source": f"Prontuário {patient_id}",
        },
    )


def test_chroma_ingestion_is_idempotent_and_filterable() -> None:
    """Stable document IDs must avoid duplicate vectors after a second ingestion."""
    # O backend local do Chroma pode manter arquivos HNSW abertos até o fim
    # do processo no Windows; o diretório temporário é descartado pelo SO.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory:
        service = ChromaService(
            embedding_service=FakeEmbeddingService(),
            persist_directory=Path(temporary_directory),
            collection_name="test_hospital_patients",
        )
        documents = [
            _document("DOC-1", "PCT-1", "Hemoglobina glicada 7.1 por cento."),
            _document("DOC-2", "PCT-2", "Glicemia de jejum 90 mg/dL."),
        ]

        first = service.add_documents(documents)
        second = service.add_documents(documents)
        filtered = service.search_by_patient("hemoglobina", "PCT-1", k=5)

        assert first.inserted == 2
        assert first.vector_count == 2
        assert second.inserted == 0
        assert second.ignored == 2
        assert second.vector_count == 2
        assert len(filtered) == 1
        assert filtered[0][0].metadata["patient_id"] == "PCT-1"
        changed = _document("DOC-1", "PCT-1", "Resultado corrigido.")
        assert service.add_documents([changed]).updated == 1
        reopened = ChromaService(FakeEmbeddingService(), Path(temporary_directory), service.collection_name)
        assert reopened.search_by_patient("resultado", "PCT-1")[0][0].page_content == changed.page_content
        import pytest
        different = FakeEmbeddingService()
        different.model_name = "other-384"
        with pytest.raises(ValueError, match="incompatível"):
            ChromaService(different, Path(temporary_directory), service.collection_name).create_or_load_vectorstore()
