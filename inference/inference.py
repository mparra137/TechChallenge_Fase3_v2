
import time
import threading
import torch

from inference.model_loader import load_medical_model

class VRAMMonitor:

    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []

    def _monitor(self):

        while not self.stop_event.wait(self.interval):

            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free, total = torch.cuda.mem_get_info(0)

            sample = {
                "allocated": allocated,
                "reserved": reserved,
                "free": free / 1024**3,
                "total": total / 1024**3
            }

            self.samples.append(sample)

            print(
                f"[VRAM] Alocada: {allocated:.2f} GB | "
                f"Reservada: {reserved:.2f} GB | "
                f"Livre: {free / 1024**3:.2f} GB"
            )

    def start(self):
        self.samples = []
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread:
            self.thread.join()


def generate_response(model, tokenizer, question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical evidence assistant specialized "
                "in clinical and scientific information. "
                "Provide accurate, clear, and evidence-based answers. "
                "Do not invent information. "
                "If you are uncertain, explicitly state your limitations. "
                "Use appropriate medical terminology. "                      
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Formatação no padrão de conversa do Llama 3.1
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenização
    input_device = model.get_input_embeddings().weight.device

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(input_device)

    input_tokens = inputs["input_ids"].shape[1]

    # Limpa estatísticas anteriores
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    monitor = VRAMMonitor(interval=1.0)

    print("\nIniciando inferência...")
    monitor.start()

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():

            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=256,
            #     do_sample=False,
            #     use_cache=True,
            #     pad_token_id=tokenizer.eos_token_id
            # )

            # Identifica o token de encerramento de turno do Llama 3
            eot_token_id = tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"            
            )

            eos_ids = [tokenizer.eos_token_id]

            if (eot_token_id is not None and eot_token_id not in eos_ids):
                eos_ids.append(eot_token_id)

            outputs = model.generate(
                **inputs,

                # Limite de geração
                max_new_tokens=256,

                # Geração determinística
                do_sample=False,

                # Controle de repetição
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,

                # Cache de atenção
                use_cache=True,

                # Encerramento correto
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id
            )

        torch.cuda.synchronize()

    finally:
        monitor.stop()

    end_time = time.perf_counter()

    # Extrai somente os tokens gerados
    generated_tokens = outputs.shape[1] - input_tokens

    response = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    elapsed = end_time - start_time

    # Estatísticas de memória
    peak_allocated = (
        torch.cuda.max_memory_allocated(0) / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(0) / 1024**3
    )

    print("\n" + "=" * 60)
    print("RESULTADO DA INFERÊNCIA")
    print("=" * 60)

    print("\nPergunta:")
    print(question)

    print("\nResposta:")
    print(response)

    print("\nMétricas:")

    print("Tokens de entrada:", input_tokens)
    print("Tokens gerados:", generated_tokens)
    print("Tempo:", round(elapsed, 2), "segundos")

    print(
        "Velocidade:",
        round(generated_tokens / elapsed, 2),
        "tokens/segundo"
    )

    print("\nConsumo máximo de VRAM:")
    print("Alocada:", round(peak_allocated, 2), "GB")
    print("Reservada:", round(peak_reserved, 2), "GB")

    return response


if __name__ == "__main__":

    model, tokenizer = load_medical_model()

    question = (
        "What is the importance of evaluating blood glucose "
        "and glycated hemoglobin (HbA1c) together? "
        "Explain the differences between these tests."
    )

    generate_response(
        model,
        tokenizer,
        question
    )
