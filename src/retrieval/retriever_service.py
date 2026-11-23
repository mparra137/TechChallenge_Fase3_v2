"""Recuperação semântica com filtros seguros de metadados."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional

from src.models.schemas import DocumentSource, PatientResolution, RetrievedDocument
from src.retrieval.patient_resolver import PatientResolver
from src.retrieval.structured import exact_query
from src.ingestion.document_loader import HospitalDocumentLoader
from src.vectorstore.chroma_service import ChromaService

logger = logging.getLogger("rag.retriever")


@dataclass(frozen=True)
class RetrievalResult:
    """Documents and patient resolution returned by one retrieval request."""

    documents: list[RetrievedDocument]
    patient: PatientResolution
    exact_answer: str | None = None


class RetrieverService:
    """Combine deterministic patient resolution with vector similarity search."""

    def __init__(self, vectorstore: ChromaService, patient_resolver: PatientResolver, default_k: int = 5,
                 loader: HospitalDocumentLoader | None = None) -> None:
        if default_k <= 0:
            raise ValueError("default_k deve ser maior que zero.")
        self.vectorstore = vectorstore
        self.patient_resolver = patient_resolver
        self.default_k = default_k
        self.loader = loader

    @staticmethod
    def _to_retrieved_document(document: object, score: float) -> RetrievedDocument:
        metadata = getattr(document, "metadata")
        source = DocumentSource(
            document_id=str(metadata["document_id"]),
            patient_id=str(metadata.get("patient_id")) if metadata.get("patient_id") else None,
            patient_name=str(metadata.get("patient_name")) if metadata.get("patient_name") else None,
            document_type=str(metadata.get("document_type")) if metadata.get("document_type") else None,
            document_date=str(metadata.get("document_date")) if metadata.get("document_date") else None,
            source=str(metadata.get("source")) if metadata.get("source") else None,
        )
        return RetrievedDocument(source=source, content=str(getattr(document, "page_content")), score=float(score))

    def retrieve(
        self,
        question: str,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        document_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> RetrievalResult:
        """Recover the most relevant documents after resolving optional patient identity."""
        if not question or not question.strip():
            raise ValueError("A pergunta não pode ser vazia.")
        limit = self.default_k if k is None else k
        if limit < 1:
            raise ValueError("K deve ser positivo.")
        patient = self.patient_resolver.resolve(patient_id=patient_id, patient_name=patient_name)
        mentioned = self.patient_resolver.from_question(question)
        if patient.status == "not_requested":
            patient = mentioned
        elif mentioned.status != "not_requested" and (
            mentioned.status != "found" or patient.patient_id != mentioned.patient_id
        ):
            raise ValueError("Identificação da pergunta diverge do filtro. Confirme o prontuário.")
        if patient.status in {"not_found", "ambiguous"}:
            return RetrievalResult(documents=[], patient=patient)
        return self._retrieve(question, patient, document_type, limit)

    def retrieve_for_patient(self, question: str, patient: PatientResolution) -> RetrievalResult:
        """Retrieve for an already resolved patient, with no global fallback."""
        if patient.status != "found" or not patient.patient_id or not question.strip():
            raise ValueError("A consulta exige um paciente resolvido e uma pergunta.")
        return self._retrieve(question, patient, None, self.default_k, scoped=True)

    def _retrieve(
        self, question: str, patient: PatientResolution, document_type: str | None, limit: int,
        *, scoped: bool = False,
    ) -> RetrievalResult:
        if self.loader:
            exact = exact_query(question, patient, document_type, self.loader, scoped=scoped)
            if exact is not None:
                answer, documents = exact
                if patient.patient_id and any(d.source.patient_id != patient.patient_id for d in documents):
                    raise ValueError("Documentos fora do escopo do prontuário.")
                return RetrievalResult(documents, patient, answer)

        metadata_filters: list[dict[str, str]] = []
        if patient.patient_id:
            metadata_filters.append({"patient_id": patient.patient_id})
        if document_type:
            metadata_filters.append({"document_type": document_type})
        metadata_filter = None
        if len(metadata_filters) == 1:
            metadata_filter = metadata_filters[0]
        elif len(metadata_filters) > 1:
            metadata_filter = {"$and": metadata_filters}

        started_at = time.perf_counter()
        results = self.vectorstore.search_documents(question, k=limit, metadata_filter=metadata_filter)
        documents = [self._to_retrieved_document(document, score) for document, score in results]
        if patient.patient_id and any(d.source.patient_id != patient.patient_id for d in documents):
            raise ValueError("Documentos fora do escopo do prontuário.")
        logger.info(
            "Recuperação concluída: documentos=%s tempo=%.3fs filtro_paciente=%s",
            len(documents),
            time.perf_counter() - started_at,
            bool(patient.patient_id),
        )
        return RetrievalResult(documents=documents, patient=patient)
