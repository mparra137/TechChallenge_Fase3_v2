"""Geração de resposta a partir de um contexto documental delimitado."""

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt


class TextChat(Protocol):
    """Chat backend capable of generating text from messages."""

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage | str:
        """Return the generated message."""
        ...


def generate_answer(llm: TextChat, question: str, context: str, *, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Return only text blocks from the chat provider's response."""
    return invoke_text(llm, [SystemMessage(content=system_prompt),
                             HumanMessage(content=build_user_prompt(question, context))])


def invoke_text(llm: TextChat, messages: Sequence[BaseMessage]) -> str:
    """Invoke a chat backend and extract its text response."""
    message = llm.invoke(messages)
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(block if isinstance(block, str) else block.get("text", "")
                          for block in content if isinstance(block, (dict, str)))
    return content.strip() if isinstance(content, str) else ""
