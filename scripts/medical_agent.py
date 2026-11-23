"""CLI do agente médico local para o acervo hospitalar sintético."""
from __future__ import annotations

import argparse
import logging
from time import perf_counter
from uuid import uuid4

from chromadb.errors import ChromaError
from huggingface_hub.errors import HfHubHTTPError

from src import config
from src.agent.graph import MedicalAgent
from src.agent.records import synthesis_response_format
from src.agent.state import AgentAnswer
from src.llm.llm_factory import get_llm
from src.llm.local_chat import LocalChat
from src.rag.components import build_rag_components
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)
_HELP = "/ajuda  comandos | /limpar  apagar contexto | /nova  nova conversa | /sair  encerrar"


def render(answer: AgentAnswer) -> None:
    """Show the validated answer and its minimal source references."""
    if answer.patient_name:
        print(answer.patient_name)
    print(answer.text)
    if answer.sources:
        print("Fontes")
        for source in answer.sources:
            print(f"- {source.document_id} | {source.date} | {source.source}")


def interactive(agent: MedicalAgent) -> None:
    """Process one turn at a time and discard memory on exit."""
    thread_id = uuid4().hex
    print("Assistente pronto. Base sintética para uso acadêmico. Digite /ajuda.")
    try:
        while True:
            try:
                entry = input("\nVocê> ").strip()
                if not entry:
                    continue
                if entry == "/sair":
                    return
                if entry == "/ajuda":
                    print(_HELP)
                elif entry in {"/limpar", "/nova"}:
                    agent.clear(thread_id)
                    if entry == "/nova":
                        thread_id = uuid4().hex
                    print("Nova conversa." if entry == "/nova" else "Contexto limpo.")
                elif entry.startswith("/"):
                    print(_HELP)
                else:
                    render(agent.ask(entry, thread_id))
            except (EOFError, KeyboardInterrupt):
                print("\nConversa encerrada.")
                return
    finally:
        agent.clear(thread_id)


def main() -> int:
    """Start local inference, connect to the persisted RAG and enter the CLI."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--log-level", choices=("WARNING", "INFO"), default="WARNING",
        help="Nível mínimo dos logs exibidos no console (padrão: %(default)s).",
    )

    args = parser.parse_args()

    configure_logging(console_level=args.log_level)

    from inference.inference import LocalInference, load_medical_model

    try:
        started = perf_counter()
        inference = LocalInference(*load_medical_model())
        logger.info("Carga do modelo: segundos=%.3f", perf_counter() - started)
    except (ImportError, HfHubHTTPError, OSError, RuntimeError, ValueError) as exc:
        logger.error("Falha ao carregar o Llama local (%s). Confira CUDA e o cache do modelo.", type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130
    retriever = None
    resolver = None
    try:
        _, store, retriever = build_rag_components(create_if_missing=False)
        resolver = retriever.patient_resolver
        if not store.get_collection_stats()["vector_count"]:
            raise ValueError("Coleção vazia.")
    except (ChromaError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Prontuários indisponíveis (%s).", type(exc).__name__)
        retriever = None
    except KeyboardInterrupt:
        return 130
    chat = LocalChat(inference, config.AGENT_ANSWER_TOKENS, config.AGENT_CONTEXT_TOKENS)
    classifier = LocalChat(inference, config.AGENT_CLASSIFICATION_TOKENS, config.AGENT_CONTEXT_TOKENS)
    record_llm = None
    try:
        if retriever is not None:
            record_llm = get_llm().bind(response_format=synthesis_response_format())
    except (RuntimeError, ValueError) as exc:
        logger.warning("Síntese documental OpenAI indisponível (%s). Confira a configuração do provedor.",
                       type(exc).__name__)
    interactive(MedicalAgent(
        chat, classifier, retriever, record_llm=record_llm, resolver=resolver,
        max_context_chars=config.RAG_MAX_CONTEXT_CHARS,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
