"""Carregamento validado dos JSONL sintéticos usados pelo RAG."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from src.utils.logger import setup_logger

logger = setup_logger("rag.document_loader")

REQUIRED_PATIENT_FIELDS = ("patient_id", "full_name")
REQUIRED_DOCUMENT_FIELDS = (
    "document_id",
    "patient_id",
    "patient_name",
    "document_type",
    "document_date",
    "source",
    "content",
)


@dataclass(frozen=True)
class IntegrityReport:
    """Resultado da validação relacional entre pacientes e documentos."""

    patient_count: int
    document_count: int
    documents_per_patient: dict[str, int]
    documents_by_type: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Indicate whether no integrity error was found."""
        return not self.errors


def _normalize_metadata(value: Any) -> str | int | float | bool:
    """Convert values to Chroma-compatible scalar metadata without losing content."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class HospitalDocumentLoader:
    """Loads the original JSONL records and maps each record to one LangChain Document."""

    def __init__(self, patients_file: Path, documents_file: Path) -> None:
        self.patients_file = patients_file
        self.documents_file = documents_file

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file while reporting its line number on malformed content."""
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}, linha {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Registro inválido em {path}, linha {line_number}: objeto esperado.")
                rows.append(row)
        return rows

    def load_patients(self) -> list[dict[str, Any]]:
        """Load patient records without changing their original structure."""
        patients = self._load_jsonl(self.patients_file)
        logger.info("Pacientes carregados: %s", len(patients))
        return patients

    def load_documents(self) -> list[dict[str, Any]]:
        """Load clinical document records without changing their original structure."""
        documents = self._load_jsonl(self.documents_file)
        logger.info("Documentos hospitalares carregados: %s", len(documents))
        return documents

    def validate_integrity(
        self,
        patients: Iterable[dict[str, Any]],
        documents: Iterable[dict[str, Any]],
        expected_patient_count: int | None = None,
        expected_document_count: int | None = None,
    ) -> IntegrityReport:
        """Validate identities, relationships and document metadata without fixed limits."""
        patient_rows = list(patients)
        document_rows = list(documents)
        errors: list[str] = []
        warnings: list[str] = []

        patient_ids: list[str] = []
        patient_names: list[str] = []
        for patient in patient_rows:
            missing = [field for field in REQUIRED_PATIENT_FIELDS if not patient.get(field)]
            if missing:
                errors.append(f"Paciente sem campos obrigatórios: {missing}.")
                continue
            patient_ids.append(str(patient["patient_id"]))
            patient_names.append(str(patient["full_name"]).casefold().strip())

        if len(patient_ids) != len(set(patient_ids)):
            errors.append("Foram encontrados patient_id duplicados.")
        if len(patient_names) != len(set(patient_names)):
            warnings.append("Foram encontrados nomes completos duplicados/homônimos.")

        known_patients = {str(patient["patient_id"]): patient for patient in patient_rows if patient.get("patient_id")}
        document_ids: list[str] = []
        document_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        for document in document_rows:
            missing = [field for field in REQUIRED_DOCUMENT_FIELDS
                       if not isinstance(document.get(field), str) or not document[field].strip()]
            if not isinstance(document.get("metadata", {}), dict):
                errors.append("Documento com metadados em formato inválido.")
            document_id = str(document.get("document_id", "<sem document_id>"))
            if missing:
                errors.append(f"Documento {document_id} sem campos obrigatórios: {missing}.")
                continue
            document_ids.append(document_id)
            patient_id = str(document["patient_id"])
            document_counter[patient_id] += 1
            type_counter[str(document["document_type"])] += 1
            if patient_id not in known_patients:
                errors.append(f"Documento {document_id} referencia paciente inexistente: {patient_id}.")
                continue
            if document["patient_name"] != known_patients[patient_id].get("full_name"):
                errors.append(f"Documento {document_id} possui patient_name inconsistente.")
            try:
                date.fromisoformat(str(document["document_date"]))
            except ValueError:
                errors.append(f"Documento {document_id} possui document_date inválida.")

        if len(document_ids) != len(set(document_ids)):
            errors.append("Foram encontrados document_id duplicados.")
        patients_without_documents = sorted(set(known_patients) - set(document_counter))
        if patients_without_documents:
            errors.append(
                f"Pacientes sem documentos associados: {', '.join(patients_without_documents[:10])}."
            )
        if expected_patient_count is not None and len(patient_rows) != expected_patient_count:
            warnings.append(
                f"Referência inicial de pacientes: {expected_patient_count}; encontrado: {len(patient_rows)}."
            )
        if expected_document_count is not None and len(document_rows) != expected_document_count:
            warnings.append(
                f"Referência inicial de documentos: {expected_document_count}; encontrado: {len(document_rows)}."
            )

        report = IntegrityReport(
            patient_count=len(patient_rows),
            document_count=len(document_rows),
            documents_per_patient=dict(document_counter),
            documents_by_type=dict(type_counter),
            errors=errors,
            warnings=warnings,
        )
        logger.info(
            "Integridade RAG validada: pacientes=%s documentos=%s válido=%s erros=%s",
            report.patient_count,
            report.document_count,
            report.valid,
            len(report.errors),
        )
        return report

    @staticmethod
    def to_langchain_documents(records: Iterable[dict[str, Any]]) -> tuple[list[Document], int]:
        """Convert each valid original clinical record into one stable LangChain Document."""
        converted: list[Document] = []
        rejected = 0
        for record in records:
            missing = [field for field in REQUIRED_DOCUMENT_FIELDS
                       if not isinstance(record.get(field), str) or not record[field].strip()]
            if missing or not isinstance(record.get("metadata", {}), dict):
                rejected += 1
                logger.warning("Documento rejeitado por campos ausentes: %s", missing)
                continue
            metadata: dict[str, str | int | float | bool] = {
                field: _normalize_metadata(record[field])
                for field in REQUIRED_DOCUMENT_FIELDS
                if field != "content"
            }
            for key, value in record.get("metadata", {}).items():
                if key not in metadata and value is not None:
                    metadata[key] = _normalize_metadata(value)
            converted.append(Document(page_content=str(record["content"]), metadata=metadata))
        logger.info("Documentos convertidos para LangChain: %s; rejeitados: %s", len(converted), rejected)
        return converted, rejected
