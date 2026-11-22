"""Build and validate patient evidence without exposing retrieved documents."""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from src.agent.guardrails import Redactor, unsafe_output
from src.agent.state import Citation
from src.models.schemas import RetrievedDocument
from src.rag.answer_generator import TextChat, generate_answer
from src.rag.context_builder import ContextBuilder, ContextBuildResult
from src.rag.prompts import SYNTHESIS_PROMPT
from src.retrieval.retriever_service import RetrievalResult
from src.retrieval.patient_resolver import normalize_patient_name

_DOCUMENT_ID = re.compile(r"\b(?:DOC|ATD|LAB|IMG|EVOL)-[A-Z0-9-]+\b", re.I)
_LAB_VALUE = re.compile(r"[^:\r\n]+: [-+]?\d+(?:[.,]\d+)? \S+(?: \S+)?")
_DELTA = re.compile(r"[-+]\d+(?:\.\d+)? \S+(?: \S+)?")
_TREND_TERMS = re.compile(
    r"\b(?:evolucao|tendencia|compar\w*|melhor\w*|pior\w*|aument\w*|diminui\w*|mudou|"
    r"trend|changed|improv\w*|worsen\w*|increas\w*|decreas\w*)\b"
)


class _Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    sources: list[str] = Field(min_length=1)


class _Synthesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    findings: list[_Finding]


def synthesis_response_format() -> dict:
    """Request the same synthesis schema from OpenAI that the agent validates locally."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "documentary_synthesis", "strict": True, "schema": _Synthesis.model_json_schema()},
    }


class RecordEvidence:
    """Use scoped, minimized documents to build verifiable answers."""

    def __init__(self, builder: ContextBuilder, redactor: Redactor) -> None:
        self.builder = builder
        self.redactor = redactor

    def _citation(self, document: RetrievedDocument) -> Citation | None:
        source = document.source
        if not source.document_date or not _DOCUMENT_ID.fullmatch(source.document_id) or not source.source:
            return None
        try:
            recorded = date.fromisoformat(source.document_date)
        except ValueError:
            return None
        origin = self.redactor.evidence(source.source, preserve_chart=source.patient_id)
        return Citation(source.document_id, recorded.isoformat(), origin) if origin else None

    def exact_answer(
        self, result: RetrievalResult, question: str, english: bool
    ) -> tuple[str, tuple[Citation, ...]]:
        """Render structured facts from sanitized documents and validated provenance."""
        parts: list[str] = []
        sources: list[Citation] = []
        for doc in result.documents:
            citation = self._citation(doc)
            content = self.redactor.evidence(doc.content, preserve_chart=doc.source.patient_id)
            if citation and content and not unsafe_output(content):
                parts.append(f"- {content} ({citation.date}) [{citation.document_id}]")
                sources.append(citation)
        if not parts:
            return "", ()
        if _TREND_TERMS.search(normalize_patient_name(question)):
            if len(sources) == len(result.documents) and _DELTA.fullmatch(result.exact_answer or ""):
                delta = result.exact_answer
                parts.append((f"Recorded numerical change: {delta}." if english else f"Variação numérica registrada: {delta}.")
                             + f" [{sources[0].document_id}] [{sources[-1].document_id}]")
                parts.append("Clinical improvement or deterioration requires professional assessment." if english
                             else "A avaliação de melhora ou piora clínica requer validação profissional.")
            else:
                parts.append("The records do not provide comparable measurements for a trend." if english
                             else "Os registros não fornecem medições comparáveis para avaliar a tendência.")
        elif len(parts) > 1 and all(_LAB_VALUE.fullmatch(self.redactor.evidence(d.content)) for d in result.documents):
            parts.append("The date has multiple documented measurements." if english
                         else "Há mais de uma medição documentada na data informada.")
        return "\n".join(parts), tuple(dict.fromkeys(sources))

    def _context(self, documents: list[RetrievedDocument]) -> tuple[ContextBuildResult, dict[str, Citation]]:
        candidates: list[RetrievedDocument] = []
        provenance: dict[str, Citation] = {}
        seen = set()
        for doc in documents:
            citation = self._citation(doc)
            content = self.redactor.evidence(_DOCUMENT_ID.sub("[DOCUMENTO]", doc.content))
            if not citation or not content or unsafe_output(content) or citation.document_id in seen:
                continue
            seen.add(citation.document_id)
            alias = f"D{len(candidates) + 1}"
            provenance[alias] = citation
            # Keep patient metadata out of the model context.
            source = doc.source.model_copy(update={
                "document_id": alias, "patient_id": None, "patient_name": None,
                "source": None, "document_date": citation.date,
                "document_type": self.redactor.evidence(doc.source.document_type or ""),
            })
            candidates.append(doc.model_copy(update={"source": source, "content": content}))
        context = self.builder.build(candidates, whole_documents=True)
        return context, {alias: provenance[alias] for alias in context.fragments}

    def synthesize(
        self, question: str, documents: list[RetrievedDocument], llm: TextChat
    ) -> tuple[str, tuple[Citation, ...]]:
        """Generate from sanitized evidence and resolve only aliases sent to the provider."""
        context, provenance = self._context(documents)
        if not provenance:
            return "", ()
        question = self.redactor.clean(_DOCUMENT_ID.sub("[DOCUMENTO]", question))
        answer = generate_answer(llm, question, context.context, system_prompt=SYNTHESIS_PROMPT)
        if answer == "EVIDENCIA_INSUFICIENTE":
            return "", ()
        fenced = re.fullmatch(r"\x60{3}(?:json)?\s*(.*?)\s*\x60{3}", answer, re.DOTALL | re.I)
        synthesis = _Synthesis.model_validate_json(fenced[1] if fenced else answer)
        parts: list[str] = []
        sources: list[Citation] = []
        for finding in synthesis.findings:
            if (unsafe_output(finding.text) or _DOCUMENT_ID.search(finding.text)
                    or re.search(r"\[D\d+\]", finding.text, re.I)):
                raise ValueError("Texto documental incompatível com o contrato de síntese.")
            text = self.redactor.clean(finding.text)
            if not text or any(alias not in provenance for alias in finding.sources):
                raise ValueError("Síntese documental sem fontes verificáveis.")
            citations = [provenance[alias] for alias in dict.fromkeys(finding.sources)]
            references = " ".join(f"[{citation.document_id}]" for citation in citations)
            parts.append(f"- {text} {references}")
            sources.extend(citations)
        return "\n".join(parts), tuple(dict.fromkeys(sources))
