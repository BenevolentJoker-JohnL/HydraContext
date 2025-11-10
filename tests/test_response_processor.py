"""
Tests for response_processor and provider_parsers modules.
"""

import unittest
from hydracontext.core.response_processor import ResponseNormalizer
from hydracontext.core.provider_parsers import (
    OpenAIParser,
    AnthropicParser,
    OllamaParser,
    UnifiedResponseParser
)


class TestOpenAIParser(unittest.TestCase):
    """Test OpenAI parser."""

    def test_parse_chat_completion(self):
        """Test parsing OpenAI chat completion."""
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        parser = OpenAIParser()
        result = parser.parse(response)

        self.assertEqual(result['content'], "Hello!")
        self.assertEqual(result['provider'], "openai")
        self.assertEqual(result['model'], "gpt-4")
        self.assertEqual(result['usage']['total_tokens'], 15)

    def test_parse_string_response(self):
        """Test parsing plain string."""
        parser = OpenAIParser()
        result = parser.parse("Plain text")
        self.assertEqual(result['content'], "Plain text")


class TestAnthropicParser(unittest.TestCase):
    """Test Anthropic parser."""

    def test_parse_message(self):
        """Test parsing Anthropic message."""
        response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Hello there!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }

        parser = AnthropicParser()
        result = parser.parse(response)

        self.assertEqual(result['content'], "Hello there!")
        self.assertEqual(result['provider'], "anthropic")
        self.assertEqual(result['usage']['prompt_tokens'], 10)
        self.assertEqual(result['usage']['completion_tokens'], 5)


class TestOllamaParser(unittest.TestCase):
    """Test Ollama parser."""

    def test_parse_response(self):
        """Test parsing Ollama response."""
        response = {
            "model": "llama2",
            "response": "Hello from Ollama!",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 15
        }

        parser = OllamaParser()
        result = parser.parse(response)

        self.assertEqual(result['content'], "Hello from Ollama!")
        self.assertEqual(result['provider'], "ollama")
        self.assertEqual(result['model'], "llama2")
        self.assertEqual(result['usage']['prompt_tokens'], 10)
        self.assertEqual(result['usage']['completion_tokens'], 15)
        self.assertEqual(result['finish_reason'], "stop")


class TestUnifiedResponseParser(unittest.TestCase):
    """Test unified parser auto-detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = UnifiedResponseParser()

    def test_detect_openai(self):
        """Test OpenAI auto-detection."""
        response = {
            "choices": [{"message": {"content": "Hi"}}],
            "usage": {}
        }
        result = self.parser.parse(response)
        self.assertEqual(result['provider'], "openai")

    def test_detect_anthropic(self):
        """Test Anthropic auto-detection."""
        response = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1}
        }
        result = self.parser.parse(response)
        self.assertEqual(result['provider'], "anthropic")

    def test_detect_ollama(self):
        """Test Ollama auto-detection."""
        response = {
            "response": "Hi",
            "done": True,
            "model": "llama2"
        }
        result = self.parser.parse(response)
        self.assertEqual(result['provider'], "ollama")

    def test_explicit_provider(self):
        """Test explicit provider specification."""
        response = {"content": "Test"}
        result = self.parser.parse(response, provider="openai")
        # Should attempt OpenAI parsing


class TestResponseNormalizer(unittest.TestCase):
    """Test ResponseNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = ResponseNormalizer()

    def test_normalize_text(self):
        """Test text normalization."""
        text = "   Hello   \n\n\n\n   World   "
        result = self.normalizer.normalize(text)
        self.assertIn('normalized_content', result)
        self.assertLess(len(result['normalized_content']), len(text))

    def test_remove_thinking_blocks(self):
        """Test thinking block removal."""
        text = "<thinking>Internal reasoning</thinking>\n\nMain content"
        result = self.normalizer.normalize(text)
        self.assertNotIn("<thinking>", result['normalized_content'])
        self.assertIn("Main content", result['normalized_content'])

    def test_normalize_json_response_ollama(self):
        """Test JSON normalization for Ollama."""
        response = {
            "model": "llama2",
            "response": "Test response",
            "done": True,
            "eval_count": 10
        }

        result = self.normalizer.normalize_json_response(response, provider="ollama")

        self.assertEqual(result['content'], "Test response")
        self.assertEqual(result['provider'], "ollama")
        self.assertIn('usage', result)

    def test_normalize_json_response_openai(self):
        """Test JSON normalization for OpenAI."""
        response = {
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 20}
        }

        result = self.normalizer.normalize_json_response(response, provider="openai")

        self.assertEqual(result['content'], "Test response")
        self.assertEqual(result['provider'], "openai")

    def test_hash_consistency(self):
        """Test hash consistency."""
        text = "Same content"
        result1 = self.normalizer.normalize(text)
        result2 = self.normalizer.normalize(text)
        self.assertEqual(result1['hash'], result2['hash'])


if __name__ == '__main__':
    unittest.main()
