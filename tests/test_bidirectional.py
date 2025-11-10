"""
Tests for bidirectional normalization.
"""

import unittest
from hydracontext.core.bidirectional import ContextNormalizer, MultiProviderNormalizer


class TestContextNormalizer(unittest.TestCase):
    """Test ContextNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = ContextNormalizer()

    def test_normalize_input(self):
        """Test input normalization."""
        messy_prompt = "   What is   AI?   \n\n\n   "
        result = self.normalizer.normalize_input(messy_prompt)

        self.assertIn('content', result)
        self.assertIn('type', result)
        self.assertIn('token_estimate', result)
        self.assertEqual(result['direction'], 'input')

    def test_normalize_output(self):
        """Test output normalization."""
        response = "Response text with   extra   spaces"
        result = self.normalizer.normalize_output(response, provider='ollama', model='llama2')

        self.assertIn('normalized_content', result)
        self.assertEqual(result['direction'], 'output')

    def test_normalize_ollama_output(self):
        """Test Ollama-specific output normalization."""
        ollama_response = {
            "model": "llama2",
            "response": "Test response",
            "done": True,
            "eval_count": 10
        }

        result = self.normalizer.normalize_ollama_output(ollama_response)

        self.assertEqual(result['content'], "Test response")
        self.assertEqual(result['provider'], 'ollama')
        self.assertEqual(result['direction'], 'output')

    def test_get_stats(self):
        """Test statistics retrieval."""
        self.normalizer.normalize_input("Test prompt")
        stats = self.normalizer.get_stats()

        self.assertIn('input_processor', stats)


class TestMultiProviderNormalizer(unittest.TestCase):
    """Test MultiProviderNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = MultiProviderNormalizer()

    def test_register_provider(self):
        """Test provider registration."""
        self.normalizer.register_provider('ollama', {'setting': 'value'})
        self.assertIn('ollama', self.normalizer.provider_configs)

    def test_normalize_input_for_provider(self):
        """Test provider-specific input normalization."""
        result = self.normalizer.normalize_input_for_provider(
            "Test prompt",
            provider='ollama'
        )

        self.assertEqual(result['provider'], 'ollama')
        self.assertIn('content', result)

    def test_normalize_outputs_from_providers(self):
        """Test normalizing outputs from multiple providers."""
        responses = [
            {
                'provider': 'ollama',
                'model': 'llama2',
                'response': 'Response from Ollama'
            },
            {
                'provider': 'openai',
                'model': 'gpt-4',
                'response': 'Response from OpenAI'
            }
        ]

        results = self.normalizer.normalize_outputs_from_providers(responses)

        self.assertEqual(len(results), 2)
        self.assertIn('normalized_content', results[0])
        self.assertIn('normalized_content', results[1])


if __name__ == '__main__':
    unittest.main()
