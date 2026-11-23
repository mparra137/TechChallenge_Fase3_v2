"""Serviço idempotente de persistência e consulta no ChromaDB."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from src.embeddings.embedding_service import EmbeddingService
from src.models.schemas import IngestionStats

logger = logging.getLogger("rag.chroma")


def _metadata_signature(metadata: dict[str, Any]) -> str:
    """Produce a stable comparison representation for scalar Chroma metadata."""
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)


class ChromaService:
    """Manage one persisted Chroma collection without deleting it on initialization."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_directory: Path,
        collection_name: str,
        *,
        create_if_missing: bool = True,
    ) -> None:
        self.embedding_service = embedding_service
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.create_if_missing = create_if_missing
        self._vectorstore: Any | None = None

    def create_or_load_vectorstore(self) -> Any:
        """Create or reopen the configured persisted collection without clearing it."""
        if self._vectorstore is not None:
            return self._vectorstore
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise RuntimeError("langchain-chroma não está instalado.") from exc
        if self.create_if_missing:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        elif not (self.persist_directory / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Índice Chroma indisponível.")
        signature = {"embedding_model": self.embedding_service.model_name, "strategy": "whole-document-v1"}
        client_options = {}
        if not self.create_if_missing:
            from chromadb.config import Settings

            client_options["client_settings"] = Settings(anonymized_telemetry=False)
        store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_service.get_embeddings(),
            persist_directory=str(self.persist_directory),
            collection_metadata=signature,
            create_collection_if_not_exists=self.create_if_missing,
            **client_options,
        )
        if store._collection.metadata != signature:
            raise ValueError("Modelo ou estratégia incompatível. Configure outra coleção em src/config.py.")
        self._vectorstore = store
        logger.info("Coleção Chroma pronta: %s em %s", self.collection_name, self.persist_directory)
        return self._vectorstore

    @staticmethod
    def _document_id(document: Document) -> str | None:
        """Read the mandatory stable vector identifier from a document's metadata."""
        value = document.metadata.get("document_id")
        return str(value) if value else None

    def _existing_records(self, identifiers: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        """Return existing document content and metadata keyed by stable identifier."""
        if not identifiers:
            return {}
        vectorstore = self.create_or_load_vectorstore()
        result = vectorstore.get(ids=identifiers, include=["documents", "metadatas"])
        return {
            identifier: (content or "", metadata or {})
            for identifier, content, metadata in zip(
                result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
            )
        }

    def add_documents(
        self,
        documents: Iterable[Document],
        total_read: int | None = None,
        rejected: int = 0,
    ) -> IngestionStats:
        """Insert new documents, update changed ones and ignore unchanged stable IDs."""
        document_list = list(documents)
        valid_documents: list[Document] = []
        invalid_count = rejected
        for document in document_list:
            if not self._document_id(document) or not document.page_content.strip():
                invalid_count += 1
                logger.warning("Documento rejeitado na ingestão por ID ou conteúdo ausente.")
                continue
            valid_documents.append(document)

        existing = self._existing_records([self._document_id(document) for document in valid_documents if self._document_id(document)])
        to_insert: list[Document] = []
        to_update: list[Document] = []
        ignored = 0
        for document in valid_documents:
            identifier = self._document_id(document)
            assert identifier is not None
            current = existing.get(identifier)
            if current is None:
                to_insert.append(document)
                continue
            old_content, old_metadata = current
            if old_content == document.page_content and _metadata_signature(old_metadata) == _metadata_signature(document.metadata):
                ignored += 1
            else:
                to_update.append(document)

        vectorstore = self.create_or_load_vectorstore()
        if to_insert:
            vectorstore.add_documents(to_insert, ids=[self._document_id(document) for document in to_insert])
        if to_update:
            self.update_documents(to_update)
        expected_ids = {doc.metadata["document_id"] for doc in valid_documents}
        stored_ids = set(vectorstore.get(include=[])["ids"])
        if not expected_ids.issubset(stored_ids):
            raise RuntimeError("Ingestão incompleta: há identificadores ausentes na coleção persistida.")

        stats = IngestionStats(
            total_read=total_read if total_read is not None else len(document_list),
            total_converted=len(valid_documents),
            inserted=len(to_insert),
            updated=len(to_update),
            ignored=ignored,
            rejected=invalid_count,
            vector_count=len(stored_ids),
        )
        logger.info("Ingestão Chroma concluída: %s", stats.model_dump())
        return stats

    def update_documents(self, documents: Iterable[Document]) -> int:
        """Explicitly update documents that already exist in the collection."""
        document_list = list(documents)
        if not document_list:
            return 0
        identifiers = [self._document_id(document) for document in document_list]
        if any(identifier is None for identifier in identifiers):
            raise ValueError("Todos os documentos atualizados devem possuir document_id.")
        self.create_or_load_vectorstore().update_documents(documents=document_list, ids=identifiers)
        logger.info("Documentos atualizados no Chroma: %s", len(document_list))
        return len(document_list)

    def search_documents(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Search semantically with an optional Chroma metadata filter."""
        if not query or not query.strip():
            raise ValueError("A consulta não pode ser vazia.")
        if k <= 0:
            raise ValueError("k deve ser maior que zero.")
        results = self.create_or_load_vectorstore().similarity_search_with_score(
            query, k=k, filter=metadata_filter
        )
        logger.info("Consulta vetorial executada: recuperados=%s", len(results))
        return results

    def search_by_patient(self, query: str, patient_id: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search only documents belonging to one resolved patient."""
        return self.search_documents(query=query, k=k, metadata_filter={"patient_id": patient_id})

    def search_by_document_type(
        self,
        query: str,
        document_type: str,
        k: int = 5,
        patient_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Search documents of one type, optionally constrained to one patient."""
        filters: list[dict[str, Any]] = [{"document_type": document_type}]
        if patient_id:
            filters.insert(0, {"patient_id": patient_id})
        metadata_filter: dict[str, Any] = filters[0] if len(filters) == 1 else {"$and": filters}
        return self.search_documents(query=query, k=k, metadata_filter=metadata_filter)

    def get_collection_stats(self) -> dict[str, Any]:
        """Return persisted vector count without exposing clinical contents."""
        vectorstore = self.create_or_load_vectorstore()
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "vector_count": int(vectorstore._collection.count()),
        }

    def clear_collection(self, confirm: bool = False) -> None:
        """Delete the collection only after an explicit caller confirmation."""
        if not confirm:
            raise ValueError("A limpeza exige confirm=True explícito.")
        vectorstore = self.create_or_load_vectorstore()
        vectorstore.delete_collection()
        self._vectorstore = None
        logger.warning("Coleção Chroma removida explicitamente: %s", self.collection_name)
