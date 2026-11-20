"""Tests for validated loading of the synthetic hospital JSONL files."""

from pathlib import Path

from src.ingestion.document_loader import HospitalDocumentLoader


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_load_and_validate_current_hospital_base() -> None:
    """The checked-in synthetic base must preserve its documented relationships."""
    loader = HospitalDocumentLoader(
        ROOT_DIR / "data" / "rag_data" / "patients.jsonl",
        ROOT_DIR / "data" / "rag_data" / "medical_documents.jsonl",
    )

    patients = loader.load_patients()
    documents = loader.load_documents()
    report = loader.validate_integrity(
        patients,
        documents,
        expected_patient_count=200,
        expected_document_count=1787,
    )

    assert report.valid
    assert report.patient_count == len(patients) > 0
    assert report.document_count == sum(report.documents_by_type.values()) == len(documents)
    assert len(report.documents_per_patient) == len(patients)


def test_conversion_keeps_one_document_per_original_record() -> None:
    """Conversion must preserve the original clinical content and stable IDs."""
    loader = HospitalDocumentLoader(
        ROOT_DIR / "data" / "rag_data" / "patients.jsonl",
        ROOT_DIR / "data" / "rag_data" / "medical_documents.jsonl",
    )
    original = loader.load_documents()[:3]

    documents, rejected = loader.to_langchain_documents(original)

    assert rejected == 0
    assert len(documents) == len(original)
    assert documents[0].metadata["document_id"] == original[0]["document_id"]
    assert documents[0].metadata["patient_id"] == original[0]["patient_id"]
    assert documents[0].page_content == original[0]["content"]


def test_rejects_orphans_duplicate_ids_and_blank_content(tmp_path):
    loader = HospitalDocumentLoader(tmp_path / "patients", tmp_path / "docs")
    row = dict(document_id="D", patient_id="other", patient_name="A", content="x",
               document_type="cadastro", source="fonte", document_date="2020-01-01")
    report = loader.validate_integrity([{"patient_id": "P", "full_name": "A"}], [row, row])
    assert not report.valid
    assert all(any(word in error for error in report.errors)
               for word in ["inexistente", "duplicados", "sem documentos"])
    assert loader.to_langchain_documents([dict(row, content="   ")]) == ([], 1)
