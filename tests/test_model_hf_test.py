"""Testes unitários para o script scripts/model_hf_test.py."""

import unittest
from unittest.mock import MagicMock, patch

from scripts.model_hf_test import (
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    HuggingFaceAdapterNotDeployableError,
    HuggingFaceClient,
    LLMClientException,
    ModelTesterRunner,
    OpenAIClient,
)


class TestHuggingFaceClient(unittest.TestCase):
    """Testes unitários para o cliente do Hugging Face."""

    def test_missing_api_key_raises_value_error(self):
        """Verifica se ValueError é levantado quando não há token configurado."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                HuggingFaceClient(api_token=None)

    @patch("requests.post")
    def test_successful_chat_completion(self, mock_post):
        """Verifica se a resposta do Hugging Face é extraída corretamente em caso de sucesso."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Biomedical answer from HF"}}]
        }
        mock_post.return_value = mock_response

        client = HuggingFaceClient(api_token="fake_hf_token")
        response = client.generate_chat_completion(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_message=DEFAULT_QUESTION,
        )

        self.assertEqual(response, "Biomedical answer from HF")
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_lora_not_supported_raises_specific_error(self, mock_post):
        """Verifica se erro 400 informando 'not a chat model' levanta HuggingFaceAdapterNotDeployableError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error":{"message":"The requested model is not a chat model."}}'
        mock_post.return_value = mock_response

        client = HuggingFaceClient(api_token="fake_hf_token")
        with self.assertRaises(HuggingFaceAdapterNotDeployableError):
            client.generate_chat_completion(
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                user_message=DEFAULT_QUESTION,
            )


class TestOpenAIClient(unittest.TestCase):
    """Testes unitários para o cliente da OpenAI."""

    def test_missing_api_key_raises_value_error(self):
        """Verifica se ValueError é levantado quando não há token OpenAI configurado."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIClient(api_key=None)

    @patch("requests.post")
    def test_successful_chat_completion(self, mock_post):
        """Verifica se a resposta da OpenAI é extraída corretamente."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Biomedical answer from OpenAI"}}]
        }
        mock_post.return_value = mock_response

        client = OpenAIClient(api_key="fake_openai_token")
        response = client.generate_chat_completion(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_message=DEFAULT_QUESTION,
        )

        self.assertEqual(response, "Biomedical answer from OpenAI")
        mock_post.assert_called_once()


class TestModelTesterRunner(unittest.TestCase):
    """Testes unitários para o orquestrador do teste de modelo."""

    def test_hf_success_no_fallback_needed(self):
        """Verifica que se HF tem sucesso, OpenAI não é acionado."""
        mock_hf = MagicMock(spec=HuggingFaceClient)
        mock_hf.generate_chat_completion.return_value = "HF Result"

        mock_openai = MagicMock(spec=OpenAIClient)

        runner = ModelTesterRunner(hf_client=mock_hf, openai_client=mock_openai)
        result = runner.run_test(
            question=DEFAULT_QUESTION,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            allow_openai_fallback=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider_used"], "huggingface")
        self.assertEqual(result["response"], "HF Result")
        mock_openai.generate_chat_completion.assert_not_called()

    def test_hf_fails_fallback_to_openai(self):
        """Verifica que falha de LoRA no HF aciona fallback para OpenAI."""
        mock_hf = MagicMock(spec=HuggingFaceClient)
        mock_hf.generate_chat_completion.side_effect = HuggingFaceAdapterNotDeployableError(
            "LoRA adapter limitation"
        )

        mock_openai = MagicMock(spec=OpenAIClient)
        mock_openai.model_name = "gpt-4o-mini"
        mock_openai.generate_chat_completion.return_value = "OpenAI Result"

        runner = ModelTesterRunner(hf_client=mock_hf, openai_client=mock_openai)
        result = runner.run_test(
            question=DEFAULT_QUESTION,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            allow_openai_fallback=True,
        )

        self.assertEqual(result["status"], "fallback_success")
        self.assertIn("openai", result["provider_used"])
        self.assertEqual(result["response"], "OpenAI Result")
        mock_openai.generate_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
