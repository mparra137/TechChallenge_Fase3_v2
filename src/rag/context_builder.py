"""Montagem limitada, deduplicada e citável do contexto recuperado."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.schemas import RetrievedDocument
from src.config import RAG_MAX_CONTEXT_TOKENS


@dataclass(frozen=True)
class ContextBuildResult:
    """Context assembled for a generation and the exact documents represented in it."""

    context: str
    documents: list[RetrievedDocument]


class ContextBuilder:
    """Build a readable context without exceeding the configured character budget."""

    def __init__(self, max_characters: int, max_tokens: int = RAG_MAX_CONTEXT_TOKENS) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters deve ser maior que zero.")
        self.max_characters = max_characters
        if max_tokens <= 0:
            raise ValueError("max_tokens deve ser positivo.")
        self.max_tokens = max_tokens

    @staticmethod
    def _header(document: RetrievedDocument) -> str:
        source = document.source
        return (
            f"[document_id: {source.document_id} | paciente: {source.patient_id or 'não informado'} "
            f"| tipo: {source.document_type or 'não informado'} "
            f"| data: {source.document_date or 'não informada'} "
            f"| fonte: {source.source or 'não informada'}]"
        )

    def build(self, documents: list[RetrievedDocument]) -> ContextBuildResult:
        """Deduplicate by document ID and append only content that fits the context limit."""
        remaining = self.max_characters
        remaining_bytes = self.max_tokens  # UTF-8 bytes bound the number of OpenAI BPE tokens.
        parts: list[str] = []
        included: list[RetrievedDocument] = []
        seen_ids: set[str] = set()
        for document in documents:
            document_id = document.source.document_id
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            header = self._header(document)
            separator = "\n\n" if parts else ""
            available_for_content = remaining - len(separator) - len(header) - 1
            if available_for_content <= 0:
                break
            content = document.content.strip()
            byte_budget = remaining_bytes - len((separator + header + "\n").encode("utf-8"))
            if byte_budget <= 0:
                break
            content = content.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore")
            if len(content) > available_for_content:
                content = content[:available_for_content].rstrip()
            if not content:
                continue
            part = f"{header}\n{content}"
            parts.append(part)
            included.append(document)
            remaining -= len(separator) + len(part)
            remaining_bytes -= len((separator + part).encode("utf-8"))
            if remaining <= 0:
                break
        return ContextBuildResult(context="\n\n".join(parts), documents=included)
