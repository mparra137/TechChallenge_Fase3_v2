"""Fábricas de provedores de LLM desacopladas do RAG."""

from src.llm.llm_factory import get_llm

__all__ = ["get_llm"]
