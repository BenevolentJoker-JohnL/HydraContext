"""
Tests for prompt_processor module.
"""

import unittest
from hydracontext.core.prompt_processor import (
    normalize_prompt,
    split_prompt,
    detect_prompt_type,
    PromptProcessor
)


class TestNormalizePrompt(unittest.TestCase):
    """Test normalize_prompt function."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        result = normalize_prompt("   Hello   World   ")
        self.assertNotIn("   ", result)
        self.assertEqual(result.strip(), "Hello   World")

    def test_normalize_line_endings(self):
        """Test line ending normalization."""
        result = normalize_prompt("Line1\r\nLine2\rLine3")
        self.assertNotIn("\r", result)
        self.assertEqual(result.count("\n"), 2)

    def test_normalize_code_blocks(self):
        """Test code block normalization."""
        input_text = "```python\n\n\ncode```"
        result = normalize_prompt(input_text)
        self.assertNotIn("\n\n\n", result)

    def test_empty_input(self):
        """Test empty input handling."""
        result = normalize_prompt("")
        self.assertEqual(result, "")


class TestDetectPromptType(unittest.TestCase):
    """Test detect_prompt_type function."""

    def test_detect_code(self):
        """Test code detection."""
        self.assertEqual(detect_prompt_type("def foo(): pass"), "code")
        self.assertEqual(detect_prompt_type("```python\ncode```"), "code")

    def test_detect_instruction(self):
        """Test instruction detection."""
        self.assertEqual(detect_prompt_type("Explain how this works"), "instruction")
        self.assertEqual(detect_prompt_type("What is AI?"), "instruction")

    def test_detect_conversation(self):
        """Test conversation detection."""
        text = "User: Hello\nAssistant: Hi there!"
        self.assertEqual(detect_prompt_type(text), "conversation")

    def test_detect_example(self):
        """Test example detection."""
        self.assertEqual(detect_prompt_type("Example: x = 5"), "example")


class TestSplitPrompt(unittest.TestCase):
    """Test split_prompt function."""

    def test_split_long_text(self):
        """Test splitting long text."""
        text = "A" * 5000
        segments = split_prompt(text, max_chars=1000)
        self.assertGreater(len(segments), 1)

    def test_short_text_no_split(self):
        """Test short text doesn't get split."""
        text = "Short text"
        segments = split_prompt(text, max_chars=1000)
        self.assertEqual(len(segments), 1)

    def test_segment_metadata(self):
        """Test segment contains metadata."""
        text = "Test content"
        segments = split_prompt(text, max_chars=1000)
        self.assertIn("content", segments[0])
        self.assertIn("length", segments[0])
        self.assertIn("id", segments[0])


class TestPromptProcessor(unittest.TestCase):
    """Test PromptProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = PromptProcessor(max_chars=1000, overlap=100)

    def test_process_single_prompt(self):
        """Test processing single prompt."""
        result = self.processor.process("Test prompt")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['type'], 'instruction')

    def test_process_batch(self):
        """Test batch processing."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = self.processor.process_batch(prompts)
        self.assertEqual(len(results), 3)

    def test_deduplication(self):
        """Test deduplication across prompts."""
        prompts = ["Same prompt", "Different prompt", "Same prompt"]
        results = self.processor.process_batch(prompts)

        # Third prompt should be marked as duplicate
        self.assertTrue(results[2][0].get('duplicate', False))

    def test_statistics(self):
        """Test statistics tracking."""
        self.processor.process("Test")
        stats = self.processor.get_statistics()
        self.assertIn('processed_count', stats)
        self.assertEqual(stats['processed_count'], 1)

    def test_reset(self):
        """Test reset functionality."""
        self.processor.process("Test")
        self.processor.reset()
        stats = self.processor.get_statistics()
        self.assertEqual(stats['processed_count'], 0)


if __name__ == '__main__':
    unittest.main()
