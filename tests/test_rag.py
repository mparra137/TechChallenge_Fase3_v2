"""Tests for context construction and RAG responses with a simulated LLM."""

from dataclasses import dataclass

from src.models.schemas import DocumentSource, PatientResolution, RetrievedDocument
from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import NO_EVIDENCE_MESSAGE, RAGService
from src.retrieval.retriever_service import RetrievalResult


def _document(document_id: str = "DOC-1") -> RetrievedDocument:
    return RetrievedDocument(
        source=DocumentSource(
            document_id=document_id,
            patient_id="PCT-000001",
            patient_name="Emanuel Cunha Cordeiro",
            document_type="exame_laboratorial",
            document_date="2026-01-01",
            source="Prontuário PCT-000001",
        ),
        content="Hemoglobina glicada: 7,1%. Resultado sintético.",
        score=0.1,
    )


@dataclass
class FakeRetriever:
    """Return a configured retrieval outcome without creating a vector database."""

    result: RetrievalResult

    def retrieve(self, **_: object) -> RetrievalResult:
        return self.result


class FakeLLM:
    """Simple LangChain-compatible fake for deterministic answer tests."""

    def invoke(self, _: object) -> str:
        return "A hemoglobina glicada documentada foi 7,1% (DOC-1)."


def test_context_builder_deduplicates_sources_and_limits_size() -> None:
    first = _document("DOC-1")
    duplicate = _document("DOC-1")
    result = ContextBuilder(max_characters=400).build([first, duplicate])

    assert len(result.documents) == 1
    assert result.context.count("DOC-1") == 1
    assert "Hemoglobina glicada" in result.context


def test_rag_returns_answer_and_only_sources_included_in_context() -> None:
    patient = PatientResolution(
        status="found", patient_id="PCT-000001", patient_name="Emanuel Cunha Cordeiro"
    )
    service = RAGService(
        retriever=FakeRetriever(RetrievalResult(documents=[_document()], patient=patient)),
        context_builder=ContextBuilder(max_characters=1000),
        llm=FakeLLM(),
    )

    response = service.answer("Qual foi a hemoglobina glicada?", patient_id="PCT-000001")

    assert response.success
    assert "7,1%" in response.answer
    assert [source.document_id for source in response.sources] == ["DOC-1"]
    assert len(response.retrieved_documents) == 1


def test_rag_does_not_call_llm_when_no_context_is_recovered() -> None:
    patient = PatientResolution(status="not_requested")
    service = RAGService(
        retriever=FakeRetriever(RetrievalResult(documents=[], patient=patient)),
        context_builder=ContextBuilder(max_characters=1000),
        llm=FakeLLM(),
    )

    response = service.answer("Qual foi o último exame?")

    assert not response.success
    assert response.answer == NO_EVIDENCE_MESSAGE
    assert not response.sources


def test_invented_source_rejected():
    from unittest.mock import Mock
    model = Mock()
    model.invoke.return_value = "Resposta [DOC-999]."
    service = RAGService(FakeRetriever(RetrievalResult([_document()], PatientResolution(status="found"))),
                         ContextBuilder(2000), model)
    response = service.answer("O que consta no registro?")
    assert not response.success and not response.sources
    assert "DOC-999" not in response.answer


def test_openai_factory_round_trip_with_mock_transport():
    import httpx
    from dataclasses import replace
    from unittest.mock import patch
    from langchain_openai import ChatOpenAI
    from src.config import get_rag_settings
    from src.llm.llm_factory import get_llm

    def complete(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"id": "test", "model": "test", "object": "chat.completion",
            "created": 1, "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Consta no registro [DOC-1]."}}]})

    with httpx.Client(transport=httpx.MockTransport(complete)) as client:
        with patch("langchain_openai.ChatOpenAI", side_effect=lambda **kw: ChatOpenAI(http_client=client, **kw)):
            model = get_llm(replace(get_rag_settings(), openai_api_key="test", openai_model="test"))
        service = RAGService(FakeRetriever(RetrievalResult([_document()], PatientResolution(status="found"))),
                             ContextBuilder(1000), model)
        assert service.answer("O que consta no registro?").success
