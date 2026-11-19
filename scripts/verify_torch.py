import torch
import bitsandbytes as bnb
import transformers
import peft
import accelerate

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Compute Capability:",
        torch.cuda.get_device_capability(0)
    )
    print(
        "VRAM total:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

    print(
        "VRAM disponível:",
        round(
            torch.cuda.mem_get_info()[0] / 1024**3,
            2
        ),
        "GB"
    )

    # Teste de operação na GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)

    print("Operação CUDA executada:", y.device)

    print("PyTorch:", torch.__version__)
    print("BitsAndBytes:", bnb.__version__)
    print("Transformers:", transformers.__version__)
    print("PEFT:", peft.__version__)
    print("Accelerate:", accelerate.__version__)

    print("BF16 suportado:", torch.cuda.is_bf16_supported())

    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))