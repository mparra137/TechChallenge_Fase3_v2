"""Schemas tipados para entradas, resultados e respostas do RAG."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """Metadados de proveniência de um documento recuperado."""

    document_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    document_type: Optional[str] = None
    document_date: Optional[str] = None
    source: Optional[str] = None


class RetrievedDocument(BaseModel):
    """Documento efetivamente retornado pelo retriever."""

    source: DocumentSource
    content: str
    score: Optional[float] = None


class PatientResolution(BaseModel):
    """Resultado determinístico de resolução de paciente."""

    status: Literal["found", "not_found", "ambiguous", "not_requested"]
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    message: Optional[str] = None


class IngestionStats(BaseModel):
    """Resumo auditável de uma operação de ingestão no vetorstore."""

    total_read: int = 0
    total_converted: int = 0
    inserted: int = 0
    updated: int = 0
    ignored: int = 0
    rejected: int = 0
    vector_count: int = 0


class RAGResponse(BaseModel):
    """Resposta pública do serviço RAG, incluindo somente fontes consultadas."""

    success: bool
    question: str
    answer: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    sources: list[DocumentSource] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    error: Optional[str] = None


class RAGState(TypedDict, total=False):
    """Contrato conceitual compatível com futura orquestração em LangGraph."""

    question: str
    patient_id: Optional[str]
    patient_name: Optional[str]
    retrieved_documents: list[RetrievedDocument]
    context: str
    answer: Optional[str]
    sources: list[DocumentSource]
    error: Optional[str]
