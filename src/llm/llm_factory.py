"""Factory do provedor de chat, preparada para extensões futuras."""

from __future__ import annotations

from typing import Any

from src import config
from src.utils.logger import setup_logger

logger = setup_logger("rag.llm_factory")


def get_llm(settings: config.RagSettings | None = None) -> Any:
    """Return the configured LangChain chat model without coupling RAG to OpenAI."""
    active_settings = settings or config.validate_rag_settings(require_llm=True)
    if active_settings.llm_provider == "openai":
        if not active_settings.openai_api_key or not active_settings.openai_model:
            raise ValueError("OPENAI_API_KEY e OPENAI_MODEL são obrigatórios para o provedor OpenAI.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("langchain-openai não está instalado.") from exc
        options: dict[str, Any] = {
            "model": active_settings.openai_model,
            "api_key": active_settings.openai_api_key,
            "temperature": 0,
        }
        if active_settings.openai_base_url:
            options["base_url"] = active_settings.openai_base_url
        logger.info("LLM OpenAI configurado: modelo=%s", active_settings.openai_model)
        return ChatOpenAI(**options)
    if active_settings.llm_provider in {"local", "groq"}:
        raise NotImplementedError(
            f"O provedor '{active_settings.llm_provider}' está reservado para adaptador futuro."
        )
    raise ValueError(f"LLM_PROVIDER inválido: {active_settings.llm_provider}.")
