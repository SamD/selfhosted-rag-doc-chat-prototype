import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Set dummy environment variables BEFORE any project imports to satisfy strict validation
os.environ["LLM_PATH"] = "/tmp/dummy.gguf"
os.environ["SUPERVISOR_LLM_PATH"] = "/tmp/supervisor.gguf"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp/dummy_e5"
os.environ["DEFAULT_DOC_INGEST_ROOT"] = "/tmp/test_docs"


# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm_setup import RemoteLlama, get_supervisor_llm
from workers.gatekeeper_logic import CHUNK0_GBNF_STR, get_llm_and_grammar


class TestLLMSetupRemote(unittest.TestCase):
    def test_remote_llama_url_sanitization(self):
        """Verify that RemoteLlama correctly sanitizes various URL formats."""
        urls = [
            ("http://1.2.3.4:11434", "http://1.2.3.4:11434/v1"),
            ("http://1.2.3.4:11434/", "http://1.2.3.4:11434/v1"),
            ("http://1.2.3.4:11434/v1", "http://1.2.3.4:11434/v1"),
            ("http://1.2.3.4:11434/v1/", "http://1.2.3.4:11434/v1"),
            ("https://api.remote.com/v1/chat/completions", "https://api.remote.com/v1"),
        ]

        for input_url, expected in urls:
            with patch("openai.OpenAI"):
                remote = RemoteLlama(base_url=input_url)
                self.assertEqual(remote.base_url, expected)

    @patch("openai.OpenAI")
    def test_remote_llama_call_passes_grammar(self, mock_openai_class):
        """Verify that RemoteLlama uses Chat API."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        remote = RemoteLlama(base_url="http://remote-server:11434")

        # Mock the chat response structure
        mock_choice = MagicMock()
        mock_choice.message.content = "normalized output"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        remote(prompt="test prompt")

        # Check if chat.completions.create was called
        self.assertTrue(mock_client.chat.completions.create.called)
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"][1]["content"], "test prompt")

    @patch("utils.llm_setup.Llama")
    @patch("utils.llm_setup.SUPERVISOR_LLM_PATH", "/tmp/local.gguf")
    def test_get_supervisor_llm_local(self, mock_llama_class):
        """Verify that get_supervisor_llm loads a local Llama instance for file paths."""
        from utils import llm_setup

        llm_setup._SUPERVISOR_LLM_CACHE = None  # Reset singleton

        with patch("os.path.exists", return_value=True):
            llm = get_supervisor_llm()
            self.assertNotIsInstance(llm, RemoteLlama)
            mock_llama_class.assert_called_once()

    @patch("utils.llm_setup.RemoteLlama")
    @patch("utils.llm_setup.SUPERVISOR_LLM_PATH", "http://remote-server:11434")
    def test_get_supervisor_llm_remote(self, mock_remote_class):
        """Verify that get_supervisor_llm returns a RemoteLlama instance for URLs."""
        from utils import llm_setup

        llm_setup._SUPERVISOR_LLM_CACHE = None  # Reset singleton

        llm = get_supervisor_llm()
        self.assertEqual(llm, mock_remote_class.return_value)
        mock_remote_class.assert_called_once_with(base_url="http://remote-server:11434")

    @patch("workers.gatekeeper_logic.get_supervisor_llm")
    def test_get_llm_and_grammar_remote(self, mock_get_supervisor):
        """Verify get_llm_and_grammar returns raw string for remote models."""
        mock_remote = MagicMock(spec=RemoteLlama)
        mock_get_supervisor.return_value = mock_remote

        llm, grammar = get_llm_and_grammar()
        self.assertEqual(llm, mock_remote)
        self.assertEqual(grammar, CHUNK0_GBNF_STR)
        self.assertIsInstance(grammar, str)

    @patch("workers.gatekeeper_logic.get_supervisor_llm")
    @patch("workers.gatekeeper_logic.LlamaGrammar")
    def test_get_llm_and_grammar_local(self, mock_grammar_class, mock_get_supervisor):
        """Verify get_llm_and_grammar returns LlamaGrammar object for local models."""
        mock_local = MagicMock()  # Should not be an instance of RemoteLlama
        mock_get_supervisor.return_value = mock_local

        llm, grammar = get_llm_and_grammar()
        self.assertEqual(llm, mock_local)
        self.assertEqual(grammar, mock_grammar_class.from_string.return_value)
        mock_grammar_class.from_string.assert_called_once_with(CHUNK0_GBNF_STR)


if __name__ == "__main__":
    unittest.main()
