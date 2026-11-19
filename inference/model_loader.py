
import os
from xml.parsers.expat import model
from opentelemetry.metrics import Counter
import torch
import sys
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftConfig, PeftModel

# Garante que a raiz do projeto esteja no sys.path para execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config

HF_TOKEN = config.HUGGINGFACE_API_KEY

ADAPTER_ID = "mparra137/llama-lora-medical-tc3-300"


def load_medical_model():

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não disponível.")

    print("=" * 60)
    print("CARREGANDO MODELO MÉDICO")
    print("=" * 60)

    # 1. Descobrir o modelo-base utilizado no treinamento
    adapter_config = PeftConfig.from_pretrained(
        ADAPTER_ID,
        token=HF_TOKEN
    )

    base_model_id = adapter_config.base_model_name_or_path

    print("Adaptador:", ADAPTER_ID)
    print("Modelo-base:", base_model_id)

    # 2. Configuração NF4
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 3. Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Carregar modelo-base quantizado
    print("\nCarregando modelo-base em 4 bits...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        # quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
        max_memory={
            0: "6GiB",
            "cpu": "12GiB"
        },
        token=HF_TOKEN,
        low_cpu_mem_usage=True
    )

    print("\nModelo-base carregado!")

    # 5. Aplicar adaptador LoRA
    print("\nAplicando adaptador LoRA...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_ID,
        token=HF_TOKEN,
        is_trainable=False
    )

    model.eval()

    model.generation_config.max_new_tokens = None
    model.generation_config.max_length = None    

    print("\nAdaptador aplicado com sucesso!")

    print("\nVerificação do modelo:")

    print("Modelo carregado:", model is not None)
    print("Modelo em avaliação:", not model.training)

    print(
        "Dispositivo dos embeddings:",
        model.get_input_embeddings().weight.device
    )

    print(
        "Parâmetros treináveis:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    from collections import Counter, defaultdict

    base_model = model.get_base_model()

    #devices = Counter(
    #    str(parameter.device)
    #    for parameter in base_model.parameters()
    #)

    devices = defaultdict(int)

    for parameter in base_model.parameters():
        devices[str(parameter.device)] += parameter.numel()
        
    print("\nDistribuição doe elementos dos parâmetros:")
    for device, count in devices.items():
        print(f"{device}: {count} elementos")

    # Informações sobre distribuição do modelo
    print("\nDistribuição dos componentes:")
    base_model = model.get_base_model()

    device_map = getattr(
        base_model,
        "hf_device_map",
        None
    )

    if device_map is not None:
        print(device_map)
    else:
        print("Mapa de dispositivos não disponível.")


    configModel = getattr(
        base_model.config,
        "quantization_config",
        None
    )
    print("\nConfiguração de quantização efetiva:")

    if configModel is not None:

        if hasattr(configModel, "to_dict"):
            configModel = configModel.to_dict()

        print("Método:", configModel.get("quant_method"))
        print("4 bits:", configModel.get("load_in_4bit"))
        print("Tipo:", configModel.get("bnb_4bit_quant_type"))
        print("Double Quant:", configModel.get("bnb_4bit_use_double_quant"))
        print("Compute dtype:", configModel.get("bnb_4bit_compute_dtype"))

    else:
        print("Configuração de quantização não encontrada.")

    print("\nConfiguração de geração original:")
    print("max_length:", model.generation_config.max_length)
    print("max_new_tokens:", model.generation_config.max_new_tokens)
    print("generation_config:", model.generation_config)


    print("\nMemória GPU:")
    print(
        "Alocada:",
        round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "GB"
    )

    print(
        "Reservada:",
        round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        "GB"
    )



    return model, tokenizer
