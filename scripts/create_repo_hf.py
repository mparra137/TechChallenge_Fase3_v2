# Instalação / atualização das dependências
# %pip install -U huggingface_hub hf_xet

from pathlib import Path
from huggingface_hub import login, HfApi
import os
import sys

# Garante que a raiz do projeto esteja no sys.path para execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config

# 1. Autenticação
HF_TOKEN = config.HUGGINGFACE_API_KEY

if not HF_TOKEN:
    raise ValueError(
        "Configure a variável de ambiente HF_TOKEN."
    )

login(token=HF_TOKEN)

# 2. Identificação do modelo
HF_USERNAME = "mparra137"

REPO_ID = (
    f"{HF_USERNAME}/llama-medical-tc3-merged-16bit"
)

# 3. Diretório do modelo merged no Google Drive
#MERGED_DIR = Path(
#    "/content/drive/MyDrive/Tech-Challenge_dataset/merged_16bit_model"
#)

MERGED_DIR :Path = config.DATA_DIR/"merged_16bit_model"  # Ajuste conforme necessário

# 4. Validação dos arquivos
assert MERGED_DIR.exists(), "Diretório do modelo não encontrado!"
assert (MERGED_DIR / "config.json").exists()
assert (MERGED_DIR / "model.safetensors.index.json").exists()

required_files = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json"
]


for filename in required_files:
    assert (MERGED_DIR / filename).exists(), (
        f"Arquivo obrigatório não encontrado: {filename}"
    )

shards = list(
    MERGED_DIR.glob("model-*.safetensors")
)

assert shards, "Nenhum arquivo de pesos encontrado!"

print(f"Diretório validado: {MERGED_DIR}")
print(f"Quantidade de shards: {len(shards)}")


# 5. Criação do repositório
api = HfApi()

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=True,  # Recomendo inicialmente manter privado
    exist_ok=True
)

print("Repositório criado/verificado!")

# 6. Upload dos arquivos
api.upload_folder(
    folder_path=str(MERGED_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload Llama 3.1 Medical Fine-Tuned Merged 16-bit"
)

print("Upload concluído!")
print(f"https://huggingface.co/{REPO_ID}")