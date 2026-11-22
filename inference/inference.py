"""Inferência conversacional e entrada experimental do Llama."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import torch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.model_loader import load_medical_model
from src import config

logger = logging.getLogger(__name__)
_GENERATION_LOCK = Lock()


class LocalInference:
    """Share model weights and serialize deterministic CUDA generations."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _encode(self, messages: Sequence[dict[str, str]]) -> Any:
        return self.tokenizer.apply_chat_template(
            list(messages), 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt", 
            return_dict=True,
        )

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        """Count the full prompt including chat delimiters."""
        return int(self._encode(messages)["input_ids"].shape[-1])

    def generate(self, messages: Sequence[dict[str, str]], output_tokens: int, context_limit: int) -> str:
        """Generate and decode only the continuation within the context limit."""
        with _GENERATION_LOCK:
            return self._generate(messages, output_tokens, context_limit)

    def choose(self, messages: Sequence[dict[str, str]], choices: Sequence[int], context_limit: int) -> int:
        """Generate exactly one allowed label with the Llama tokenizer."""
        labels = [self.tokenizer.encode(str(choice), add_special_tokens=False) for choice in choices]
        if any(len(label) != 1 for label in labels):
            raise ValueError("O tokenizer deve representar cada opção com um único token.")
        allowed = [label[0] for label in labels]
        with _GENERATION_LOCK:
            return int(self._generate(messages, 1, context_limit, allowed))

    def _generate(
        self, 
        messages: Sequence[dict[str, str]], 
        output_tokens: int, 
        context_limit: int,
        allowed: list[int] | None = None,
    ) -> str:
        inputs = self._encode(messages)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if output_tokens < 1 or input_tokens + output_tokens > context_limit:
            raise ValueError("A conversa excede o orçamento de tokens.")
        device = self.model.get_input_embeddings().weight.device
        gpu = device if device.type == "cuda" else torch.device("cuda:0")
        eos_ids = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.get_vocab().get("<|eot_id|>")
        if eot is not None and eot not in eos_ids:
            eos_ids.append(eot)
        torch.cuda.synchronize(gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
        started = perf_counter()
        inputs = inputs.to(device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=output_tokens, 
                do_sample=False, 
                # Controle de repetição
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                #
                use_cache=True,
                eos_token_id=eos_ids, 
                pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=(lambda _batch, _ids: allowed) if allowed is not None else None,
            )
        continuation = output[0, input_tokens:].tolist()
        torch.cuda.synchronize(gpu)
        free, _ = torch.cuda.mem_get_info(gpu)
        logger.info(
            "Geração: entrada=%s saída=%s segundos=%.3f pico_mib=%.0f reservado_mib=%.0f livres_mib=%.0f",
            input_tokens, len(continuation), perf_counter() - started,
            torch.cuda.max_memory_allocated(gpu) / 2**20, torch.cuda.max_memory_reserved(gpu) / 2**20, free / 2**20,
        )
        return self.tokenizer.decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()

    @staticmethod
    def release_cached_memory() -> None:
        """Release unused blocks before a reduced retry after CUDA OOM."""
        with _GENERATION_LOCK:
            torch.cuda.empty_cache()


def generate_response(model: Any, tokenizer: Any, question: str) -> str:
    """Run the original one-question experiment and render its result."""
    answer = LocalInference(model, tokenizer).generate(
        [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": question}],
        config.AGENT_ANSWER_TOKENS, config.AGENT_CONTEXT_TOKENS,
    )
    print(answer)
    return answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_response(*load_medical_model(), "What is the difference between blood glucose and HbA1c?")
