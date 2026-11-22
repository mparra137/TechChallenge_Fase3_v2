"""LangChain message adapter over the shared local generator."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage

if TYPE_CHECKING:
    from inference.inference import LocalInference


class LocalChat:
    """Provide separate generation profiles over the same model and tokenizer."""

    def __init__(self, inference: LocalInference, output_tokens: int, context_limit: int) -> None:
        self.inference = inference
        self.output_tokens = output_tokens
        self.context_limit = context_limit

    @staticmethod
    def _messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        result = []
        for message in messages:
            if message.type not in roles or not isinstance(message.content, str):
                raise ValueError("O Llama local aceita apenas mensagens de texto da conversa.")
            result.append({"role": roles[message.type], "content": message.content})
        return result

    def count_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """Count the templated prompt with the local tokenizer."""
        return self.inference.count_tokens(self._messages(messages))

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a chat continuation using this profile."""
        return AIMessage(content=self.inference.generate(
            self._messages(messages), self.output_tokens, self.context_limit
        ))

    def choose(self, messages: Sequence[BaseMessage], choices: Sequence[int]) -> int:
        """Choose one supplied label without generating free-form clinical claims."""
        return self.inference.choose(self._messages(messages), choices, self.context_limit)

    def release_cached_memory(self) -> None:
        """Release free allocator blocks following a failed generation."""
        self.inference.release_cached_memory()
