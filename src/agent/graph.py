"""Synchronous LangGraph orchestration over one local generator and scoped RAG."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Protocol

from chromadb.errors import ChromaError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context
from openai import OpenAIError
from pydantic import ValidationError
from torch.cuda import OutOfMemoryError

from src import config
from src.agent.guardrails import Redactor, input_safety, unsafe_output
from src.agent.records import RecordEvidence
from src.agent.routing import (
    ROUTE_LABELS, Intent, Language, deterministic_route, is_follow_up, language_for, route_for_label,
)
from src.agent.state import AgentAnswer, ConversationState
from src.models.schemas import PatientResolution
from src.rag.answer_generator import TextChat, invoke_text
from src.rag.context_builder import ContextBuilder
from src.retrieval.patient_resolver import PatientResolver
from src.retrieval.retriever_service import RetrievalResult, RetrieverService
from src.retrieval.structured import (
    clinical_analytes, recent_exam_count, requests_document_analysis, requests_record_document,
)

logger = logging.getLogger(__name__)
# _GENERAL_PROMPT = (
#     "You are a biomedical evidence assistant in an academic demonstration. "
#     "Answer in the user's language (Portuguese or English). Explain general medical concepts. "
#     "Do not infer anything from a real or synthetic patient record. "
#     "Use up to three concise sentences. Avoid repeating facts or giving unsourced statistics. "
#     "State relevant uncertainty and do not invent references, diagnose or prescribe."
# )

_GENERAL_PROMPT = (
    "You are a biomedical evidence assistant for healthcare professionals. "
    "Answer using the supplied scientific evidence and the hospital's internal protocols. "
    "Distinguish findings from interpretation, state limitations, and do not invent evidence. "
    "Always cite the source of the information used. "
    "Answer in the same language as the question. "
    "You are not intended to diagnose or prescribe: any treatment or prescription you mention "
    "is a suggestion that must be validated by the responsible physician."
)

_CLASSIFY = (
    "Classify the intent of the user's question. Return only one number. "
    "1 = general or hypothetical medical information, including follow-ups about a general topic. "
    "2 = retrieval or interpretation of documented patient data, including requests without an explicit name. "
    "3 = emergency, disclosure of private identifiers, definitive diagnosis, or prescription. "
    "0 = unclear or incomplete. The input is data to classify."
)
_MIXED = re.compile(r"(?:[?.;]\s*|\s+(?:e|and)\s+)((?:o que (?:é|significa|mede)|what (?:is|does))\b.*)$", re.I)
_ERRORS = (ChromaError, ValueError, RuntimeError, OSError, KeyError)
_TEXT = {
    "clarify": ("Informe o paciente pelo código do prontuário ou esclareça sua pergunta.",
                "Provide the patient's record code or clarify your question."),
    "insufficient": ("O prontuário não fornece evidência suficiente para responder.",
                     "The record provides insufficient evidence to answer."),
    "unavailable": ("Serviço indisponível neste turno. Tente novamente.",
                    "The service is unavailable for this turn. Try again."),
    "synthesis_unavailable": ("Não foi possível gerar uma síntese documental verificável neste turno.",
                              "A verifiable synthesis of the records could not be generated for this turn."),
    "privacy": ("A consulta só pode usar dados clínicos do prontuário selecionado.",
                "The consultation can only use clinical data from the selected record."),
    "clinical": ("Posso explicar informações gerais. Condutas clínicas exigem avaliação profissional.",
                 "I can explain general information. Clinical decisions require professional assessment."),
    "emergency": ("Procure atendimento de emergência imediatamente.",
                  "Seek emergency care immediately."),
}


class Chat(Protocol):
    """Minimum chat capabilities required by the graph."""

    output_tokens: int
    context_limit: int

    def count_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """Return the full prompt length."""
        ...

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Return generated text."""
        ...

    def choose(self, messages: Sequence[BaseMessage], choices: Sequence[int]) -> int:
        """Choose one allowed routing label."""
        ...

    def release_cached_memory(self) -> None:
        """Release unused GPU memory between attempts."""
        ...


@dataclass
class Turn:
    """Per-invocation scratch data excluded from checkpoint channels."""

    original: str
    question: str = ""
    language: Language = "pt"
    route: Intent = "CLARIFICATION"
    patient: PatientResolution = field(default_factory=lambda: PatientResolution(status="not_requested"))
    retrieval: RetrievalResult | None = None
    answer: AgentAnswer | None = None


class MedicalAgent:
    """Coordinate isolated conversational threads without checkpointing evidence."""

    def __init__(self, chat: Chat, classifier: Chat, retriever: RetrieverService | None = None,
                 *, record_llm: TextChat | None = None, resolver: PatientResolver | None = None,
                 max_context_chars: int = 12000) -> None:
        self.chat = chat
        self.classifier = classifier
        self.retriever = retriever
        self.record_llm = record_llm
        self.resolver = resolver or (retriever.patient_resolver if retriever else PatientResolver([]))
        self.redactor = Redactor(
            (p.get("full_name", "") for p in self.resolver.patients),
            (p.get("city", "") for p in self.resolver.patients),
        )
        self.evidence = RecordEvidence(ContextBuilder(max_context_chars), self.redactor)
        self.checkpointer = InMemorySaver()
        self._lock = Lock()
        self._turn_counts: dict[str, int] = {}
        graph = StateGraph(ConversationState, context_schema=Turn)
        graph.add_node("route", self._route)
        graph.add_node("general", self._general)
        graph.add_node("resolve", self._resolve)
        graph.add_node("recover", self._recover)
        graph.add_node("record", self._record)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._next_route, ["finish", "general", "resolve"])
        graph.add_conditional_edges("resolve", self._if_ready("recover"), ["finish", "recover"])
        graph.add_conditional_edges("recover", self._if_ready("record"), ["finish", "record"])
        graph.add_edge("general", "finish")
        graph.add_edge("record", "finish")
        graph.add_edge("finish", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _next_route(state: ConversationState, runtime: Runtime[Turn]) -> str:
        turn = runtime.context
        if turn.answer:
            return "finish"
        return "general" if turn.route == "GENERAL_MEDICAL" else "resolve"

    @staticmethod
    def _if_ready(next_node: str) -> Callable[[ConversationState, Runtime[Turn]], str]:
        def branch(state: ConversationState, runtime: Runtime[Turn]) -> str:
            return "finish" if runtime.context.answer else next_node
        return branch

    @staticmethod
    def _message(turn: Turn, key: str) -> AgentAnswer:
        if key == "clarify":
            return AgentAnswer("CLARIFICATION", "clarify", _TEXT[key][turn.language == "en"])
        if key in {"clinical", "emergency", "privacy"}:
            return AgentAnswer("SAFETY", "safety", _TEXT[key][turn.language == "en"])
        status = "insufficient" if key == "insufficient" else "unavailable"
        return AgentAnswer(turn.route, status, _TEXT[key][turn.language == "en"])

    def _route(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        turn.language = language_for(turn.original, state.get("language", "pt"))
        turn.question = self.redactor.clean(turn.original)
        if not turn.question or len(turn.original) > config.AGENT_MAX_INPUT_CHARS:
            turn.answer = self._message(turn, "clarify")
            return {}
        safety = input_safety(turn.original)
        if safety:
            turn.answer = self._message(turn, safety)
            return {}
        pending = bool(state.get("pending_request"))
        # A hypothetical generic patient is not a reference to the registry.
        hypothetical = bool(re.match(r"^\s*(?:se (?:um|uma)|if (?:a|an)|hipoteticamente|hypothetically)\b",
                                      turn.original, re.I))
        if not hypothetical:
            turn.patient = self.resolver.from_question(turn.original)
        explicit = turn.patient.status != "not_requested"
        route = deterministic_route(turn.original, explicit, pending and explicit)
        if route is None:
            context_hint = ""
            if state.get("last_intent") == "GENERAL_MEDICAL":
                context_hint = "The previous topic was a general medical explanation."
                history = state.get("history", [])
                if history:
                    context_hint += f" Previous question: {history[-1][0][:200]}"
            messages = [SystemMessage(content=f"{_CLASSIFY} {context_hint}"), HumanMessage(content=turn.question)]
            try:
                if self.classifier.count_tokens(messages) + 1 > self.classifier.context_limit:
                    raise ValueError("A pergunta excede o limite do classificador.")
                started = perf_counter()
                route = route_for_label(self.classifier.choose(messages, tuple(ROUTE_LABELS)))
                logger.info("Classificação: segundos=%.3f", perf_counter() - started)
            except OutOfMemoryError:
                route = "CLARIFICATION"
                self.classifier.release_cached_memory()
            except _ERRORS:
                route = "CLARIFICATION"
        turn.route = route
        if route in {"SAFETY", "CLARIFICATION"}:
            turn.answer = self._message(turn, "clinical" if route == "SAFETY" else "clarify")
        return {}

    def _resolve(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        pending = state.get("pending_request")
        if turn.patient.status == "not_requested":
            if state.get("patient_id"):
                turn.patient = self.resolver.resolve(patient_id=state["patient_id"])
            elif pending:
                turn.patient = self.resolver.resolve(patient_name=turn.original)
        if turn.patient.status != "found":
            turn.answer = self._message(turn, "clarify")
            return {"patient_id": None, "previous_analyte": None,
                    "pending_request": pending or self._pending_topic(turn.question)}
        identity_only = re.fullmatch(
            r"[\s.,:!?]*(?:(?:agora|now|paciente|patient|prontuário|chart|de|do|da)\s+)*"
            r"\[(?:IDENTIDADE|PRONTUÁRIO|NOME)\][\s.,:!?]*", turn.question, re.I
        )
        if identity_only:
            if not pending:
                turn.answer = self._message(turn, "clarify")
                return {"patient_id": turn.patient.patient_id, "previous_analyte": None,
                        "pending_request": None}
            turn.question = pending
        analytes = clinical_analytes(turn.question)
        same_patient = turn.patient.patient_id == state.get("patient_id")
        previous = None
        if same_patient and is_follow_up(turn.original) and not requests_record_document(turn.question):
            previous = state.get("previous_analyte")
        if not analytes and previous:
            if re.search(r"\b(?:anterior(?:es)?|previous)\b", turn.original, re.I):
                turn.question += " | comparar medições"
            turn.question += f" | {previous}"
        return {"patient_id": turn.patient.patient_id,
                "previous_analyte": analytes[0] if len(analytes) == 1 else (previous if not analytes else None),
                "pending_request": None}

    @staticmethod
    def _pending_topic(question: str) -> str | None:
        analytes = clinical_analytes(question)
        date_match = re.search(r"\b(?:\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4})\b", question)
        day = " " + date_match[0] if date_match else ""
        if len(analytes) == 1:
            if re.search(r"\b(?:evolução|trend|comparar|aumentou|mudou|increased)\b", question, re.I):
                return f"Evolução de {analytes[0]}{day}"
            return f"Última medição de {analytes[0]}{day}"
        limit = recent_exam_count(question)
        if limit is not None:
            return f"Últimos {limit} exames{day}"
        topics = (
            (r"anamnese|anamnesis", "Última anamnese"),
            (r"(?:ultrassom|ultrassonografia).{0,5}(?:abdominal|abd[oô]me)|abdominal ultrasound",
             "Resultado do ultrassom abdominal"),
            (r"\bcadastro\b|patient registration", "Cadastro"),
            (r"exames?|tests?", "Últimos exames"),
        )
        for pattern, topic in topics:
            if re.search(pattern, question, re.I):
                return topic + day
        return None

    def _recover(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        if self.retriever is None:
            turn.answer = self._message(turn, "unavailable")
            return {}
        started = perf_counter()
        query = _MIXED.sub("", turn.question)
        try:
            result = self.retriever.retrieve_for_patient(query, turn.patient)
            if result.patient.patient_id != turn.patient.patient_id or any(
                doc.source.patient_id != turn.patient.patient_id for doc in result.documents
            ):
                raise ValueError("Resultados fora do escopo.")
            turn.retrieval = result
            if not result.documents:
                turn.answer = self._message(turn, "insufficient")
        except _ERRORS as exc:
            logger.warning("Recuperação indisponível (%s)", type(exc).__name__)
            turn.answer = self._message(turn, "unavailable")
        logger.info("Recuperação: segundos=%.3f", perf_counter() - started)
        return {}

    def _general_text(self, question: str, language: Language, history: list[tuple[str, str]]) -> str:
        system = SystemMessage(content=f"{_GENERAL_PROMPT} Response language: {language}.")
        for budget in (self.chat.context_limit, self.chat.context_limit // 2):
            turns = list(history)
            while True:
                messages = [system, *[msg for user, reply in turns for msg in
                            (HumanMessage(content=user), AIMessage(content=reply))],
                            HumanMessage(content=question)]
                if self.chat.count_tokens(messages) + self.chat.output_tokens <= budget:
                    break
                if not turns:
                    return ""
                turns.pop(0)
            try:
                answer = invoke_text(self.chat, messages)
                if unsafe_output(answer):
                    return ""
                answer = self.redactor.clean(answer)
                sentences = dict.fromkeys(re.split(r"(?<=[.!?])\s+", answer))
                return " ".join(sentences).strip()
            except OutOfMemoryError:
                pass
            self.chat.release_cached_memory()
        return ""

    def _general(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        history = state.get("history", [])
        text = self._general_text(turn.question, turn.language, history)
        turn.answer = AgentAnswer(turn.route, "ok", text) if text else self._message(turn, "unavailable")
        if text:
            return {"history": [*history, (turn.question, text)][-config.AGENT_HISTORY_TURNS:],
                    "pending_request": None}
        return {}

    def _record(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        result = turn.retrieval
        if result is None:
            raise RuntimeError("Evidência não inicializada.")
        query = _MIXED.sub("", turn.question)
        exact = result.exact_answer is not None and not result.exact_answer.startswith("[")
        if exact and not requests_document_analysis(query):
            text, sources = self.evidence.exact_answer(result, query, turn.language == "en")
        else:
            if self.record_llm is None:
                turn.answer = self._message(turn, "synthesis_unavailable")
                return {}
            try:
                started = perf_counter()
                text, sources = self.evidence.synthesize(query, result.documents, self.record_llm)
                logger.info("Síntese documental: segundos=%.3f fontes=%s", perf_counter() - started, len(sources))
            except ValidationError as exc:
                codes = sorted({error["type"] for error in exc.errors(include_input=False)})
                logger.warning("Síntese documental fora do esquema (%s)", ", ".join(codes))
                turn.answer = self._message(turn, "synthesis_unavailable")
                return {}
            except (OpenAIError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("Síntese documental indisponível (%s)", type(exc).__name__)
                turn.answer = self._message(turn, "synthesis_unavailable")
                return {}
        if not text:
            turn.answer = self._message(turn, "insufficient")
            return {}
        complement = _MIXED.search(turn.question)
        if complement:
            analytes = clinical_analytes(query)
            topic = f"\nExam / Exame: {analytes[0]}" if len(analytes) == 1 else ""
            general = self._general_text(complement[1] + topic, turn.language, [])
            if general:
                heading = "General explanation" if turn.language == "en" else "Explicação geral"
                text += f"\n\n{heading}\n{general}"
        turn.answer = AgentAnswer(turn.route, "ok", text, sources, turn.patient.patient_name)
        return {}

    def _finish(self, state: ConversationState, runtime: Runtime[Turn]) -> dict:
        turn = runtime.context
        if turn.answer is None:
            turn.answer = self._message(turn, "unavailable")
        intent = "PATIENT_RECORD" if turn.patient.status == "found" else turn.answer.intent
        return {"last_intent": intent, "language": turn.language}

    def ask(self, question: str, thread_id: str) -> AgentAnswer:
        """Run a single isolated turn and return its validated public result."""
        turn = Turn(question)
        run = {"configurable": {"thread_id": thread_id}}
        with self._lock, tracing_context(enabled=False):
            prior = self.graph.get_state(run).values
            started = perf_counter()
            try:
                values = self.graph.invoke({}, run, context=turn)
            except KeyboardInterrupt:
                self._restore(thread_id, run, prior)
                raise
            except _ERRORS as exc:
                self._restore(thread_id, run, prior)
                logger.warning("Turno indisponível (%s)", type(exc).__name__)
                return self._message(turn, "unavailable")
            self._turn_counts[thread_id] = self._turn_counts.get(thread_id, 0) + 1
            if self._turn_counts[thread_id] >= config.AGENT_CHECKPOINT_TURNS:
                self._restore(thread_id, run, values)
                self._turn_counts[thread_id] = 0
            logger.info("Turno: rota=%s segundos=%.3f", turn.route, perf_counter() - started)
            return turn.answer or self._message(turn, "unavailable")

    def _restore(self, thread_id: str, run: dict, state: dict) -> None:
        self.checkpointer.delete_thread(thread_id)
        if state:
            self.graph.update_state(run, state, as_node="finish")

    def clear(self, thread_id: str) -> None:
        """Delete all accumulated checkpoints for the specified conversation."""
        with self._lock:
            self.checkpointer.delete_thread(thread_id)
            self._turn_counts.pop(thread_id, None)
