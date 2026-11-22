"""Shared composition of the hospital retrieval components."""
from __future__ import annotations

from pathlib import Path

from src import config
from src.embeddings.embedding_service import get_embedding_service
from src.ingestion.document_loader import HospitalDocumentLoader
from src.retrieval.patient_resolver import PatientResolver
from src.retrieval.retriever_service import RetrieverService
from src.vectorstore.chroma_service import ChromaService


def build_rag_components(
    persist_dir: Path | None = None,
    patients_path: Path | None = None,
    documents_path: Path | None = None,
    *,
    create_if_missing: bool = True,
) -> tuple[HospitalDocumentLoader, ChromaService, RetrieverService]:
    """Assemble retrieval without selecting or initializing an LLM provider."""
    settings = config.validate_rag_settings(require_llm=False)
    loader = HospitalDocumentLoader(patients_path or settings.patients_file, documents_path or settings.documents_file)
    embeddings = get_embedding_service(
        settings.embedding_model, settings.embedding_device,
        local_files_only=not create_if_missing,
    )
    store = ChromaService(
        embeddings, persist_dir or settings.chroma_persist_directory,
        settings.chroma_collection_name, create_if_missing=create_if_missing,
    )
    return loader, store, RetrieverService(store, PatientResolver(loader.load_patients()), settings.retriever_k, loader)
