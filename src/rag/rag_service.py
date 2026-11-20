"""Orquestração do fluxo RAG sem acoplamento ao provedor de LLM."""

from __future__ import annotations

import time
import re
from openai import OpenAIError
from typing import Any, Optional


from src.llm.llm_factory import get_llm
from src.models.schemas import DocumentSource, RAGResponse
from src.rag.context_builder import ContextBuilder
from src.rag.answer_generator import generate_answer
from src.retrieval.retriever_service import RetrieverService
from src.utils.logger import setup_logger

logger = setup_logger("rag.service")

NO_EVIDENCE_MESSAGE = (
    "Não localizei informações hospitalares suficientes para responder com segurança à pergunta."
)


class RAGService:
    """Retrieve evidence, build context, generate an answer and expose exact sources."""

    def __init__(
        self,
        retriever: RetrieverService,
        context_builder: ContextBuilder,
        llm: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm

    @staticmethod
    def _sources(documents: list[Any]) -> list[DocumentSource]:
        """Return unique provenance entries only for context documents actually used."""
        sources: list[DocumentSource] = []
        seen_ids: set[str] = set()
        for document in documents:
            if document.source.document_id not in seen_ids:
                sources.append(document.source)
                seen_ids.add(document.source.document_id)
        return sources

    def answer(
        self,
        question: str,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        document_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> RAGResponse:
        """Answer a question only when there is relevant retrieved hospital context."""
        if not question or not question.strip():
            return RAGResponse(
                success=False,
                question=question,
                answer="A pergunta não pode ser vazia.",
                error="Pergunta vazia.",
            )
        try:
            retrieval = self.retriever.retrieve(
                question=question,
                patient_id=patient_id,
                patient_name=patient_name,
                document_type=document_type,
                k=k,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("Falha na recuperação RAG: %s", exc)
            return RAGResponse(success=False, question=question, answer=NO_EVIDENCE_MESSAGE, error=str(exc))

        resolved_patient = retrieval.patient
        if resolved_patient.status == "ambiguous":
            return RAGResponse(
                success=False,
                question=question,
                answer="Há mais de um paciente com esse nome. Informe o código do prontuário.",
                sources=[],
                error=resolved_patient.message,
            )
        if resolved_patient.status == "not_found":
            return RAGResponse(
                success=False,
                question=question,
                answer="Não localizei o paciente informado.",
                error=resolved_patient.message,
            )

        if retrieval.exact_answer:
            return RAGResponse(success=True, question=question, answer=retrieval.exact_answer,
                               patient_id=resolved_patient.patient_id, patient_name=resolved_patient.patient_name,
                               sources=self._sources(retrieval.documents), retrieved_documents=retrieval.documents)
        context_result = self.context_builder.build(retrieval.documents)
        if not context_result.documents:
            return RAGResponse(
                success=False,
                question=question,
                answer=NO_EVIDENCE_MESSAGE,
                patient_id=resolved_patient.patient_id,
                patient_name=resolved_patient.patient_name,
                error="Nenhum documento relevante foi recuperado.",
            )

        try:
            llm = self.llm or get_llm()
            started_at = time.perf_counter()
            answer = generate_answer(llm, question, context_result.context)
            if answer == "EVIDENCIA_INSUFICIENTE":
                return RAGResponse(success=False, question=question, answer=NO_EVIDENCE_MESSAGE,
                                   patient_id=resolved_patient.patient_id, patient_name=resolved_patient.patient_name,
                                   retrieved_documents=context_result.documents, error="Evidência insuficiente.")
            if not answer:
                raise RuntimeError("O LLM retornou uma resposta vazia.")
            known = {doc.source.document_id for doc in context_result.documents}
            cited = set(re.findall(r"\b(?:DOC|ATD|LAB|IMG|EVOL)-[A-Z0-9-]+\b", answer, re.I))
            if cited - known:
                raise ValueError("A resposta citou documentos fora do contexto.")
            used = [doc for doc in context_result.documents if doc.source.document_id in cited]
            if not used:
                raise ValueError("A resposta não identificou evidência documental.")
            logger.info("Resposta RAG gerada em %.3fs", time.perf_counter() - started_at)
        except (OpenAIError, RuntimeError, ValueError, OSError) as exc:
            logger.error("Falha ao gerar resposta RAG (%s)", type(exc).__name__)
            return RAGResponse(
                success=False,
                question=question,
                answer="Não foi possível gerar uma resposta a partir dos documentos recuperados.",
                patient_id=resolved_patient.patient_id,
                patient_name=resolved_patient.patient_name,
                retrieved_documents=context_result.documents,
                error="Falha do provedor ou citação não verificável.",
            )

        return RAGResponse(
            success=True,
            question=question,
            answer=answer,
            patient_id=resolved_patient.patient_id,
            patient_name=resolved_patient.patient_name,
            sources=self._sources(used),
            retrieved_documents=context_result.documents,
        )
