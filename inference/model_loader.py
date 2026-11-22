"""Carregamento do Llama e do adaptador PEFT para inferência local."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

logger = logging.getLogger(__name__)


def load_medical_model() -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load one instance of the configured model and tokenizer on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não disponível.")

    adapter_id = config.MEDICAL_ADAPTER_ID
    if config.MEDICAL_LOCAL_FILES_ONLY and not Path(adapter_id).is_dir():
        # PEFT otherwise checks the Hub for remote adapter weights even with local_files_only.
        adapter_id = snapshot_download(adapter_id, local_files_only=True)
    hub = {"token": config.HUGGINGFACE_API_KEY, "local_files_only": config.MEDICAL_LOCAL_FILES_ONLY}
    base_id = PeftConfig.from_pretrained(adapter_id, **hub).base_model_name_or_path
    base_config = AutoConfig.from_pretrained(base_id, **hub)
    quantization = getattr(base_config, "quantization_config", None)
    if quantization is None:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.float16,
        )
        options = {"quantization_config": quantization}
    else:
        quantization = BitsAndBytesConfig.from_dict(quantization)
        if not quantization.load_in_4bit or quantization.bnb_4bit_quant_type != "nf4":
            raise ValueError("A base deve usar quantização NF4.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            quantization.bnb_4bit_compute_dtype = torch.float16
        base_config.quantization_config = quantization.to_dict()
        options = {}

    tokenizer = AutoTokenizer.from_pretrained(base_id, **hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, 
        config=base_config, 
        dtype=torch.float16, 
        device_map="auto",
        max_memory=config.MEDICAL_DEVICE_MEMORY, 
        low_cpu_mem_usage=True, 
        **hub, 
        **options,
    )
    model = PeftModel.from_pretrained(base, adapter_id, is_trainable=False, **hub)
    model.eval()
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None
    dtypes = {str(layer.compute_dtype) for layer in base.modules() if getattr(layer, "compute_dtype", None)}
    devices = {str(parameter.device) for parameter in base.parameters()}
    
    logger.info("Llama: 4-bit=%s tipo=%s compute=%s dispositivos=%s alocada_mib=%.0f",
                getattr(base, "is_loaded_in_4bit", False), quantization.bnb_4bit_quant_type,
                sorted(dtypes), sorted(devices),
                torch.cuda.memory_allocated() / 2**20)
    
    return model, tokenizer
