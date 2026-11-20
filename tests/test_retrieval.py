"""Regressões de escopo, agregação e resolução dentro da pergunta."""

from unittest.mock import Mock

import pytest

from src.retrieval.patient_resolver import PatientResolver
from src.retrieval.retriever_service import RetrieverService


@pytest.fixture
def retriever() -> RetrieverService:
    patients = [{"patient_id": "PCT-000001", "full_name": "João Almeida"}]
    loader, store = Mock(), Mock()
    loader.load_patients.return_value = patients
    store.search_documents.return_value = []
    return RetrieverService(store, PatientResolver(patients), loader=loader)


@pytest.mark.parametrize("mention", ["pct-000001", "João Almeida", "paciente Joao"])
def test_scope_from_question(retriever, mention):
    result = retriever.retrieve(f"Quais exames de {mention}?", document_type="anamnese")
    assert result.patient.patient_id == "PCT-000001"
    assert retriever.vectorstore.search_documents.call_args.kwargs["metadata_filter"] == {
        "$and": [{"patient_id": "PCT-000001"}, {"document_type": "anamnese"}]
    }


def test_unknown_code_is_never_global(retriever):
    assert retriever.retrieve("Exames de PCT-999999?").patient.status == "not_found"
    retriever.vectorstore.search_documents.assert_not_called()


def test_counts_use_complete_registry(retriever):
    assert "1 pacientes" in retriever.retrieve("Quantos pacientes há na base?").exact_answer
    retriever.vectorstore.search_documents.assert_not_called()


def test_unsupported_aggregate_refuses_sampling(retriever):
    with pytest.raises(ValueError, match="agregação"):
        retriever.retrieve("Quais pacientes têm colesterol elevado?")
    retriever.vectorstore.search_documents.assert_not_called()


def test_global_glucose_uses_documented_reference(retriever):
    retriever.loader.load_documents.return_value = [
        {"document_id": "LAB-1", "patient_id": "P1", "content": "Laudo sintético.",
         "metadata": {"results": [{"name": "Glicemia de jejum", "unit": "mg/dL", "value": 105,
                                   "reference": "70–99 mg/dL"}]}}]
    result = retriever.retrieve("Quais pacientes possuem registros de glicemia elevada?")
    assert "P1" in result.exact_answer
    assert result.documents[0].source.document_id == "LAB-1"
    retriever.vectorstore.search_documents.assert_not_called()
