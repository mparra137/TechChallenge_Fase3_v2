
import torch
import bitsandbytes as bnb

print("=" * 60)
print("VALIDAÇÃO DO BITSANDBYTES")
print("=" * 60)

print("BitsAndBytes:", bnb.__version__)
print("CUDA disponível:", torch.cuda.is_available())

# Verifica a disponibilidade de operações CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA não está disponível.")

# Testa uma camada Linear de 4 bits
try:
    layer = bnb.nn.Linear4bit(
        input_features=128,
        output_features=64,
        bias=False,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4"
    )

    layer = layer.to("cuda")

    # Executa uma operação real
    x = torch.randn(
        2, 128,
        dtype=torch.float16,
        device="cuda"
    )

    output = layer(x)

    torch.cuda.synchronize()

    print("\nTeste NF4 executado com sucesso!")
    print("Dispositivo:", output.device)
    print("Formato:", output.dtype)
    print("Dimensões:", output.shape)

except Exception as e:
    print("\nFalha no teste BitsAndBytes:")
    raise e

finally:
    del layer
    if "x" in locals():
        del x
    if "output" in locals():
        del output

    torch.cuda.empty_cache()

print("\nValidação concluída.")
