"""Script para teste e consumo do modelo fine-tuned no Hugging Face.

Consome o modelo biomédico fine-tuned hospedado em:
https://huggingface.co/mparra137/llama-lora-medical-tc3-300

O script segue estritamente as diretrizes de Clean Code, SOLID e tratamento de exceções
definidas no AGENTS.md, utilizando autenticação via .env (HUGGINGFACE_API_KEY / OPEN_AI_API_KEY).
"""

from __future__ import annotations

import abc
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Garante que a raiz do projeto esteja no sys.path para importações internas
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import setup_logger

# Carrega variáveis de ambiente do .env na raiz do projeto
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configurações globais e valores padrão
DEFAULT_MODEL_REPO_ID = "mparra137/llama-lora-medical-tc3-300"
DEFAULT_MODEL_URL = f"https://huggingface.co/{DEFAULT_MODEL_REPO_ID}"

DEFAULT_SYSTEM_PROMPT = (
    "You are a biomedical evidence assistant for healthcare professionals. "
    "Answer using the supplied scientific evidence and the hospital's internal protocols. "
    "Distinguish findings from interpretation, state limitations, and do not invent evidence. "
    "Always cite the source of the information used. "
    "Answer in the same language as the question. "
    "You are not intended to diagnose or prescribe: any treatment or prescription you mention "
    "is a suggestion that must be validated by the responsible physician."
)

DEFAULT_QUESTION = "What are the treatments for Guillain-Barr Syndrome ?"


class LLMClientException(Exception):
    """Exceção base para erros em chamadas de clientes LLM."""


class HuggingFaceAdapterNotDeployableError(LLMClientException):
    """Lançada quando a API Serverless do HF rejeita o repositório por ser adaptador LoRA."""


class BaseLLMClient(abc.ABC):
    """Interface abstrata para clientes de inferência LLM (SOLID - DIP/ISP)."""

    @abc.abstractmethod
    def generate_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Gera resposta para uma conversa no formato Chat Completion.

        Args:
            system_prompt: Diretrizes e papel do sistema.
            user_message: Pergunta ou instrução do usuário.
            temperature: Grau de aleatoriedade da geração.
            max_tokens: Quantidade máxima de tokens na resposta.

        Returns:
            str: Conteúdo textual gerado pelo modelo.

        Raises:
            LLMClientException: Se houver falha na requisição ou resposta do modelo.
        """
        raise NotImplementedError


class HuggingFaceClient(BaseLLMClient):
    """Cliente para consumo de modelos na plataforma Hugging Face via Router / Inference Endpoints."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_REPO_ID,
        api_token: Optional[str] = None,
        custom_endpoint_url: Optional[str] = None,
        timeout_seconds: int = 45,
    ) -> None:
        """Inicializa o cliente do Hugging Face.

        Args:
            model_id: Identificador do repositório no HF Hub (ex: mparra137/llama-lora-medical-tc3-300).
            api_token: Token de acesso do Hugging Face. Se None, lê de HUGGINGFACE_API_KEY ou HF_TOKEN.
            custom_endpoint_url: URL de Dedicated Inference Endpoint (opcional).
            timeout_seconds: Tempo limite da requisição HTTP em segundos.
        """
        self.model_id = model_id
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.custom_endpoint_url = custom_endpoint_url or os.getenv("HF_ENDPOINT_URL")
        self.timeout_seconds = timeout_seconds

        if not self.api_token:
            raise ValueError(
                "Chave de API do Hugging Face não encontrada. "
                "Defina HUGGINGFACE_API_KEY ou HF_TOKEN no arquivo .env."
            )

    def generate_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Envia mensagens para o modelo no Hugging Face.

        Tenta inicialmente a API compatível com OpenAI do Hugging Face Router
        (https://router.huggingface.co/v1/chat/completions) ou endpoint dedicado.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # Se houver endpoint dedicado configurado (TGI / Inference Endpoints)
        target_url = self.custom_endpoint_url or "https://router.huggingface.co/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                target_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.RequestException as err:
            raise LLMClientException(f"Erro de conexão com o Hugging Face: {err}") from err

        if response.status_code == 200:
            data = response.json()
            try:
                content: str = data["choices"][0]["message"]["content"]
                return content.strip()
            except (KeyError, IndexError) as err:
                raise LLMClientException(
                    f"Formato inesperado na resposta do Hugging Face: {data}"
                ) from err

        # Analisa respostas de erro detalhadas do Hugging Face
        error_text = response.text
        if response.status_code == 400 and (
            "not a chat model" in error_text
            or "Model not supported by provider" in error_text
            or "model_not_supported" in error_text
        ):
            raise HuggingFaceAdapterNotDeployableError(
                f"O repositório '{self.model_id}' contém adaptadores LoRA (PEFT) "
                f"treinados a partir de base 4-bit (unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit). "
                f"A infraestrutura Serverless gratuita do Hugging Face não executa adaptadores LoRA "
                f"sem um Inference Endpoint dedicado ou merge prévio dos pesos. Detalhes: {error_text}"
            )

        raise LLMClientException(
            f"Hugging Face API retornou HTTP {response.status_code}: {error_text}"
        )


class OpenAIClient(BaseLLMClient):
    """Cliente para a API da OpenAI utilizado como fallback ou baseline de comparação."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: int = 45,
    ) -> None:
        """Inicializa o cliente OpenAI.

        Args:
            api_key: Chave da OpenAI. Se None, lê de OPEN_AI_API_KEY no .env.
            model_name: Nome do modelo (ex: gpt-4o-mini). Se None, lê de OPEN_AI_MODEL.
            timeout_seconds: Tempo limite da requisição HTTP em segundos.
        """
        self.api_key = api_key or os.getenv("OPEN_AI_API_KEY")
        self.model_name = model_name or os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError(
                "Chave de API da OpenAI não encontrada. "
                "Defina OPEN_AI_API_KEY no arquivo .env."
            )

    def generate_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Envia mensagens para a API Chat Completions da OpenAI."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.RequestException as err:
            raise LLMClientException(f"Erro de conexão com a OpenAI: {err}") from err

        if response.status_code != 200:
            raise LLMClientException(
                f"OpenAI API retornou HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError) as err:
            raise LLMClientException(
                f"Formato inesperado na resposta da OpenAI: {data}"
            ) from err


class ModelTesterRunner:
    """Orquestrador do teste de consumo do modelo fine-tuned."""

    def __init__(
        self,
        hf_client: Optional[HuggingFaceClient] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """Injeta dependências de clientes de LLM."""
        self.logger = setup_logger(name="model_hf_test")
        self.hf_client = hf_client
        self.openai_client = openai_client

    def run_test(
        self,
        question: str,
        system_prompt: str,
        allow_openai_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Executa o teste de pergunta e resposta contra os provedores configurados.

        Args:
            question: Pergunta a ser enviada ao modelo.
            system_prompt: Prompt do sistema (biomédico).
            allow_openai_fallback: Se True, tenta OpenAI caso HF falhe por limitação de LoRA Serverless.

        Returns:
            Dict[str, Any]: Resultado consolidado da execução com metadados e resposta.
        """
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO TESTE DE CONSUMO DO MODELO BIOMÉDICO")
        self.logger.info("=" * 80)
        self.logger.info("Modelo Alvo: %s", DEFAULT_MODEL_REPO_ID)
        self.logger.info("URL Hugging Face: %s", DEFAULT_MODEL_URL)
        self.logger.info("Pergunta: %s", question)
        self.logger.info("System Prompt: %s", system_prompt[:120] + "...")

        result_payload: Dict[str, Any] = {
            "model_repo": DEFAULT_MODEL_REPO_ID,
            "model_url": DEFAULT_MODEL_URL,
            "question": question,
            "system_prompt": system_prompt,
            "provider_used": None,
            "status": "pending",
            "response": None,
            "elapsed_seconds": 0.0,
            "diagnostics": [],
        }

        # 1. Tentativa com Hugging Face
        start_time = time.time()
        hf_success = False

        if self.hf_client is not None:
            self.logger.info("Enviando requisição para a API do Hugging Face...")
            try:
                response = self.hf_client.generate_chat_completion(
                    system_prompt=system_prompt,
                    user_message=question,
                )
                elapsed = time.time() - start_time
                self.logger.info("✅ Resposta recebida com sucesso via Hugging Face em %.2fs", elapsed)
                result_payload["provider_used"] = "huggingface"
                result_payload["status"] = "success"
                result_payload["response"] = response
                result_payload["elapsed_seconds"] = round(elapsed, 2)
                hf_success = True
            except HuggingFaceAdapterNotDeployableError as hf_lora_err:
                self.logger.warning("⚠️ Limitação da API Serverless Hugging Face detectada:")
                self.logger.warning("%s", hf_lora_err)
                result_payload["diagnostics"].append(str(hf_lora_err))
            except LLMClientException as hf_err:
                self.logger.error("❌ Falha na chamada da API Hugging Face: %s", hf_err)
                result_payload["diagnostics"].append(str(hf_err))

        # 2. Se Hugging Face falhar e fallback estiver autorizado
        if not hf_success and allow_openai_fallback:
            if self.openai_client is None:
                try:
                    self.openai_client = OpenAIClient()
                except ValueError as val_err:
                    self.logger.error("Chave da OpenAI não configurada para fallback: %s", val_err)

            if self.openai_client is not None:
                self.logger.info("Iniciando execução de fallback via OpenAI (%s)...", self.openai_client.model_name)
                fb_start = time.time()
                try:
                    openai_response = self.openai_client.generate_chat_completion(
                        system_prompt=system_prompt,
                        user_message=question,
                    )
                    fb_elapsed = time.time() - fb_start
                    self.logger.info("✅ Resposta recebida com sucesso via OpenAI em %.2fs", fb_elapsed)
                    result_payload["provider_used"] = f"openai ({self.openai_client.model_name})"
                    result_payload["status"] = "fallback_success"
                    result_payload["response"] = openai_response
                    result_payload["elapsed_seconds"] = round(fb_elapsed, 2)
                except LLMClientException as fb_err:
                    self.logger.error("❌ Falha também na chamada de fallback da OpenAI: %s", fb_err)
                    result_payload["status"] = "all_failed"
                    result_payload["diagnostics"].append(f"OpenAI error: {fb_err}")
            else:
                result_payload["status"] = "hf_failed_no_fallback"
        elif not hf_success:
            result_payload["status"] = "hf_failed"

        return result_payload


def parse_arguments() -> argparse.Namespace:
    """Configura e analisa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Teste de inferência do modelo fine-tuned no Hugging Face."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Pergunta a ser enviada ao modelo.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_REPO_ID,
        help="Repositório no Hugging Face (padrão: mparra137/llama-lora-medical-tc3-300).",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "hf", "openai"],
        default="auto",
        help="Provedor de inferência: 'auto' (tenta HF e faz fallback para OpenAI se necessário), 'hf' (apenas HF), 'openai' (apenas OpenAI).",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="URL customizada de Dedicated Inference Endpoint no Hugging Face (opcional).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada principal para execução do teste de modelo."""
    args = parse_arguments()
    logger = setup_logger(name="model_hf_test")

    hf_client: Optional[HuggingFaceClient] = None
    openai_client: Optional[OpenAIClient] = None

    if args.provider in ["auto", "hf"]:
        try:
            hf_client = HuggingFaceClient(
                model_id=args.model_id,
                custom_endpoint_url=args.endpoint_url,
            )
        except ValueError as err:
            logger.error("Erro ao configurar HuggingFaceClient: %s", err)
            if args.provider == "hf":
                sys.exit(1)

    if args.provider in ["auto", "openai"]:
        try:
            openai_client = OpenAIClient()
        except ValueError as err:
            logger.warning("Aviso sobre OpenAIClient: %s", err)
            if args.provider == "openai":
                sys.exit(1)

    allow_fallback = (args.provider == "auto")
    runner = ModelTesterRunner(hf_client=hf_client, openai_client=openai_client)

    results = runner.run_test(
        question=args.question,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        allow_openai_fallback=allow_fallback,
    )

    # Exibe resumo formatado para o usuário
    print("\n" + "=" * 80)
    print("📊 RESULTADO DO TESTE DE MODELO")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Provedor Utilizado: {results['provider_used']}")
    print(f"Tempo de Execução: {results['elapsed_seconds']}s")
    print(f"Pergunta: {results['question']}")
    print("-" * 80)
    print("RESPOSTA GERADA:")
    print("-" * 80)
    if results["response"]:
        print(results["response"])
    else:
        print("[Nenhuma resposta foi obtida]")
    print("-" * 80)

    if results["diagnostics"]:
        print("\n🔍 DIAGNÓSTICO E NOTAS TÉCNICAS:")
        for idx, diag in enumerate(results["diagnostics"], 1):
            print(f" [{idx}] {diag}")

        print("\n💡 NOTA DE ARQUITETURA SOBRE MODELOS LORA NO HUGGING FACE:")
        print(
            " O repositório 'mparra137/llama-lora-medical-tc3-300' armazena adaptadores LoRA (PEFT)\n"
            " treinados via Unsloth sobre o modelo 4-bit 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'.\n"
            " Para consumi-lo diretamente via infraestrutura do Hugging Face, você tem 2 caminhos:\n"
            "   1. Dedicado: Criar um Hugging Face Inference Endpoint (TGI com suporte a LoRA)\n"
            "      e passar a URL via '--endpoint-url' ou variável HF_ENDPOINT_URL.\n"
            "   2. Merged Model: Fazer 'model.merge_and_unload()' no notebook e subir os pesos\n"
            "      completos do modelo (16-bit / GGUF / AWQ) para o Hugging Face Hub.\n"
            "   3. GPU Local / Colab: Carregar com 'PeftModel.from_pretrained()' conforme\n"
            "      demonstrado na célula 16/19 do notebook 'colab_finetuning.ipynb'."
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
