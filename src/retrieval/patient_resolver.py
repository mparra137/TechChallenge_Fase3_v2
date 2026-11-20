"""Resolução segura de pacientes por identificador ou nome."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from src.models.schemas import PatientResolution
from src.utils.logger import setup_logger

logger = setup_logger("rag.patient_resolver")


def normalize_patient_name(value: str) -> str:
    """Normalize casing, accents and whitespace for deterministic name matching."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_accents.casefold().split())


class PatientResolver:
    """Resolve patients without silently selecting between ambiguous name matches."""

    def __init__(self, patients: Iterable[dict[str, Any]]) -> None:
        self.patients = [dict(patient) for patient in patients]
        self._by_id = {
            str(patient["patient_id"]).casefold(): patient
            for patient in self.patients
            if patient.get("patient_id")
        }

    @staticmethod
    def _candidate(patient: dict[str, Any]) -> dict[str, str]:
        """Return the minimal non-sensitive candidate representation for disambiguation."""
        return {
            "patient_id": str(patient["patient_id"]),
            "full_name": str(patient.get("full_name", "")),
        }

    @classmethod
    def _found(cls, patient: dict[str, Any]) -> PatientResolution:
        return PatientResolution(
            status="found",
            patient_id=str(patient["patient_id"]),
            patient_name=str(patient.get("full_name", "")),
            candidates=[cls._candidate(patient)],
        )

    def resolve(
        self,
        patient_id: str | None = None,
        patient_name: str | None = None,
    ) -> PatientResolution:
        """Resolve by exact ID, exact normalized name, then unambiguous partial name."""
        if not patient_id and not patient_name:
            return PatientResolution(status="not_requested")

        resolved_by_id: dict[str, Any] | None = None
        if patient_id:
            resolved_by_id = self._by_id.get(patient_id.strip().casefold())
            if resolved_by_id is None:
                return PatientResolution(
                    status="not_found",
                    message=f"Nenhum paciente encontrado para o código informado: {patient_id}.",
                )

        if patient_name:
            normalized_query = normalize_patient_name(patient_name)
            if not normalized_query:
                return PatientResolution(status="not_found", message="Nome do paciente vazio.")
            exact = [
                patient
                for patient in self.patients
                if normalize_patient_name(str(patient.get("full_name", ""))) == normalized_query
            ]
            matches = exact or [
                patient
                for patient in self.patients
                if re.search(r"\b" + re.escape(normalized_query),
                             normalize_patient_name(str(patient.get("full_name", ""))))
            ]
            if resolved_by_id is not None:
                if resolved_by_id not in matches:
                    return PatientResolution(
                        status="not_found",
                        message="O código e o nome informados identificam pacientes diferentes.",
                    )
                return self._found(resolved_by_id)
            if not matches:
                return PatientResolution(
                    status="not_found",
                    message=f"Nenhum paciente encontrado para o nome informado: {patient_name}.",
                )
            if len(matches) > 1:
                logger.info("Resolução ambígua de paciente: candidatos=%s", len(matches))
                return PatientResolution(
                    status="ambiguous",
                    candidates=[self._candidate(patient) for patient in matches],
                    message="Mais de um paciente corresponde ao nome informado. Solicite desambiguação.",
                )
            return self._found(matches[0])

        assert resolved_by_id is not None
        return self._found(resolved_by_id)

    def from_question(self, question: str) -> PatientResolution:
        """Resolve explicit chart codes and complete names embedded in a question."""
        query = normalize_patient_name(question)
        codes = set(re.findall(r"\b(?:pct|prt)[ -]?\d+\b", query))
        names = {
            patient["full_name"] for patient in self.patients
            if re.search(r"\b" + re.escape(normalize_patient_name(patient["full_name"])) + r"\b", query)
        }
        if len(codes) > 1 or len(names) > 1:
            return PatientResolution(status="ambiguous", message="Informe um paciente por consulta.")
        if codes or names:
            return self.resolve(next(iter(codes), None), next(iter(names), None))
        mention = re.search(r"\b(?:paciente|prontuario)\s+([\w -]+)", query)
        if mention:
            name = re.split(r"\s+(?:em|no|na|com|tem|teve|apresentou|possui|realizou)\b", mention[1])[0]
            return self.resolve(patient_name=name.strip())
        return PatientResolution(status="not_requested")
