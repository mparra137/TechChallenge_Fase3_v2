"""Módulo de configurações e constantes globais do projeto."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Diretórios base
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LOGS_DIR: Path = BASE_DIR / "logs"

# Garante que os diretórios necessários existam
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Carrega variáveis do arquivo .env
ENV_FILE_PATH: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH)

# Configurações de logging
LOG_FILE_PATH: Path = LOGS_DIR / "execucao.log"

# Configurações do Hugging Face
HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

# --- Fonte 1: MedQuAD (perguntas e respostas de saúde do NIH) ---
HUGGINGFACE_DATASET_NAME: str = "lavita/MedQuAD"
USE_MEDQUAD: bool = True

# --- Fonte 2: PubMedQA (perguntas clínicas com evidência de artigos PubMed) ---
# Sugerido no enunciado do Tech Challenge Fase 3.
PUBMEDQA_DATASET_NAME: str = "qiaojin/PubMedQA"
USE_PUBMEDQA: bool = True
# 'pqa_artificial' (~211 mil, gerado automaticamente) alimenta o treino por amostragem.
PUBMEDQA_TRAIN_CONFIG: str = "pqa_artificial"
PUBMEDQA_TRAIN_SAMPLE_SIZE: int = int(os.getenv("PUBMEDQA_TRAIN_SAMPLE_SIZE", "10000"))
# 'pqa_labeled' (~1 mil, rotulado por especialistas) fica reservado para avaliação.
PUBMEDQA_EVAL_CONFIG: str = "pqa_labeled"
# Limite de caracteres do contexto (abstract) incluído na pergunta, para controlar o tamanho do prompt.
PUBMEDQA_MAX_CONTEXT_CHARS: int = 3000

# --- Fonte 3: Base sintética do hospital (protocolos, FAQ de médicos, laudos, receitas) ---
USE_SYNTHETIC: bool = True
SYNTHETIC_CASES_PER_PROTOCOL: int = int(os.getenv("SYNTHETIC_CASES_PER_PROTOCOL", "120"))
SYNTHETIC_RAW_PATH: Path = RAW_DATA_DIR / "synthetic_hospital_raw.jsonl"

# --- Fonte 4: Base sintética para RAG (scripts/hospital_rag.py) ---
RAG_DATA_DIR: Path = DATA_DIR / "rag_data"
RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAG_DEFAULT_N_PATIENTS: int = 200
RAG_DEFAULT_SEED: int = 20261111
RAG_JSONL_PATH: Path = RAG_DATA_DIR / "medical_documents.jsonl"
RAG_PATIENTS_PATH: Path = RAG_DATA_DIR / "patients.jsonl"
RAG_REPORT_PATH: Path = RAG_DATA_DIR / "validation_report.json"
RAG_CHROMA_DIR: Path = BASE_DIR / "chroma_db"
RAG_COLLECTION_NAME: str = "hospital_sintetico"
RAG_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Caminhos dos arquivos de saída de fine-tuning
TRAIN_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_train.jsonl"
VAL_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_val.jsonl"
EVAL_PUBMEDQA_JSONL_PATH: Path = PROCESSED_DATA_DIR / "eval_pubmedqa_labeled.jsonl"
DATASET_SUMMARY_PATH: Path = PROCESSED_DATA_DIR / "dataset_summary.json"
DATASET_MANIFEST_PATH: Path = PROCESSED_DATA_DIR / "dataset_manifest.csv"

# Arquivos de descarte (auditoria)
DISCARDED_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded.json"  # MedQuAD (nulos/vazios)
DISCARDED_PUBMEDQA_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded_pubmedqa.json"
DISCARDED_CURATION_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded_curation.json"

# Curadoria: filtros de tamanho (em caracteres)
MIN_QUESTION_CHARS: int = 10
MIN_ANSWER_CHARS: int = 20
MAX_ANSWER_CHARS: int = 6000
# Acrescenta a fonte ao final de cada resposta, para o modelo aprender a citar (explainability).
APPEND_SOURCE_CITATION: bool = True

# Prompt do Sistema para o fine-tuning biomédico
SYSTEM_PROMPT: str = (
    "You are a biomedical evidence assistant for healthcare professionals. "
    "Answer using the supplied scientific evidence and the hospital's internal protocols. "
    "Distinguish findings from interpretation, state limitations, and do not invent evidence. "
    "Always cite the source of the information used. "
    "Answer in the same language as the question. "
    "You are not intended to diagnose or prescribe: any treatment or prescription you mention "
    "is a suggestion that must be validated by the responsible physician. "
)

# Parâmetros de separação de dataset
DEFAULT_TRAIN_RATIO: float = 0.9
RANDOM_SEED: int = 42


# Parâmetros de operação do RAG. O ambiente fornece somente credenciais.
LLM_PROVIDER = "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Modelo previamente configurado no projeto.
OPENAI_BASE_URL: Optional[str] = None
EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
EMBEDDING_DEVICE = "cpu"
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "database" / "chroma"
CHROMA_COLLECTION_NAME = "hospital_patients"
RETRIEVER_K = 5
PATIENTS_FILE = RAG_PATIENTS_PATH
DOCUMENTS_FILE = RAG_JSONL_PATH
RAG_MAX_CONTEXT_CHARS = 12000
RAG_MAX_CONTEXT_TOKENS = 4096
RAG_EXPECTED_PATIENT_COUNT = 200
RAG_EXPECTED_DOCUMENT_COUNT = 1787
LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class RagSettings:
    """Configurações imutáveis consumidas pelos serviços do RAG."""

    llm_provider: str
    openai_api_key: Optional[str] = field(repr=False)
    openai_model: Optional[str]
    openai_base_url: Optional[str]
    embedding_model: str
    embedding_device: str
    chroma_persist_directory: Path
    chroma_collection_name: str
    retriever_k: int
    patients_file: Path
    documents_file: Path
    max_context_chars: int
    log_level: str


def get_rag_settings() -> RagSettings:
    """Return the centralized, environment-derived RAG settings."""
    return RagSettings(
        llm_provider=LLM_PROVIDER,
        openai_api_key=OPENAI_API_KEY,
        openai_model=OPENAI_MODEL,
        openai_base_url=OPENAI_BASE_URL,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
        chroma_collection_name=CHROMA_COLLECTION_NAME,
        retriever_k=RETRIEVER_K,
        patients_file=PATIENTS_FILE,
        documents_file=DOCUMENTS_FILE,
        max_context_chars=RAG_MAX_CONTEXT_CHARS,
        log_level=LOG_LEVEL,
    )


def validate_rag_settings(require_llm: bool = False) -> RagSettings:
    """Validate required settings without requiring an LLM for retrieval-only actions."""
    settings = get_rag_settings()
    if min(settings.retriever_k, settings.max_context_chars) <= 0:
        raise ValueError("K e orçamento de contexto devem ser positivos em src/config.py.")
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL não pode ser vazio.")
    if not settings.chroma_collection_name:
        raise ValueError("CHROMA_COLLECTION_NAME não pode ser vazio.")
    if require_llm:
        if settings.llm_provider != "openai":
            raise ValueError(
                f"LLM_PROVIDER '{settings.llm_provider}' ainda não é suportado. Use 'openai'."
            )
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY é obrigatória para gerar respostas com LLM.")
        if not settings.openai_model:
            raise ValueError("OPENAI_MODEL é obrigatório para gerar respostas com LLM.")
    return settings
