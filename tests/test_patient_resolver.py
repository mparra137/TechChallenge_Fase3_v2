"""Tests for deterministic and accent-insensitive patient resolution."""

from src.retrieval.patient_resolver import PatientResolver


def _patients() -> list[dict[str, str]]:
    return [
        {"patient_id": "PCT-000001", "full_name": "João da Silva"},
        {"patient_id": "PCT-000002", "full_name": "Joana da Silva"},
        {"patient_id": "PCT-000003", "full_name": "Maria Oliveira"},
    ]


def test_resolves_exact_code_with_priority() -> None:
    resolution = PatientResolver(_patients()).resolve(patient_id="pct-000001")

    assert resolution.status == "found"
    assert resolution.patient_id == "PCT-000001"


def test_resolves_exact_name_ignoring_accents_and_case() -> None:
    resolution = PatientResolver(_patients()).resolve(patient_name=" joao DA silva ")

    assert resolution.status == "found"
    assert resolution.patient_id == "PCT-000001"


def test_partial_ambiguous_name_requires_disambiguation() -> None:
    resolution = PatientResolver(_patients()).resolve(patient_name="jo")

    assert resolution.status == "ambiguous"
    assert {candidate["patient_id"] for candidate in resolution.candidates} == {
        "PCT-000001",
        "PCT-000002",
    }


def test_unknown_patient_is_not_found() -> None:
    resolution = PatientResolver(_patients()).resolve(patient_name="Paciente Inexistente")

    assert resolution.status == "not_found"
