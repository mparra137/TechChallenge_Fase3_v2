"""Geração de resposta a partir de um contexto documental delimitado."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt


def generate_answer(llm: Any, question: str, context: str) -> str:
    """Return only text blocks from the chat provider's response."""
    message = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                          HumanMessage(content=build_user_prompt(question, context))])
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(block if isinstance(block, str) else block.get("text", "")
                          for block in content if isinstance(block, (dict, str)))
    return content.strip() if isinstance(content, str) else ""
