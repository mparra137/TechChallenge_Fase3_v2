"""Checkpointed fields and the public chat result."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from src.agent.routing import Intent, Language


class ConversationState(TypedDict, total=False):
    """Only minimized conversation fields are stored by LangGraph."""

    history: list[tuple[str, str]]
    patient_id: str | None
    pending_request: str | None
    previous_analyte: str | None
    last_intent: Intent
    language: Language


@dataclass(frozen=True)
class Citation:
    """Verifiable public reference to one synthetic document."""

    document_id: str
    date: str
    source: str


@dataclass(frozen=True)
class AgentAnswer:
    """Validated output rendered by the CLI."""

    intent: Intent
    status: Literal["ok", "clarify", "insufficient", "safety", "unavailable"]
    text: str
    sources: tuple[Citation, ...] = ()
    patient_name: str | None = None
