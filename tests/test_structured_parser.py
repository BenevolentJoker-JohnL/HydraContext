"""
Tests for structured_parser module.
"""

import unittest
from hydracontext.core.structured_parser import (
    StructuredParser,
    FidelityLevel,
    parse_to_json,
    json_to_text
)


class TestStructuredParser(unittest.TestCase):
    """Test StructuredParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = StructuredParser()

    def test_parse_heading(self):
        """Test heading parsing."""
        text = "# Main Title\n\nContent here"
        result = self.parser.parse(text)

        headings = [b for b in result['blocks'] if b['type'] == 'heading']
        self.assertEqual(len(headings), 1)
        self.assertEqual(headings[0]['content'], "Main Title")
        self.assertEqual(headings[0]['level'], 1)

    def test_parse_code_block(self):
        """Test code block parsing."""
        text = "```python\nprint('hello')\n```"
        result = self.parser.parse(text)

        code_blocks = [b for b in result['blocks'] if b['type'] == 'code_block']
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]['language'], 'python')
        self.assertIn("print", code_blocks[0]['content'])

    def test_parse_list(self):
        """Test list parsing."""
        text = "- Item 1\n- Item 2\n- Item 3"
        result = self.parser.parse(text)

        lists = [b for b in result['blocks'] if b['type'] == 'list']
        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0]['item_count'], 3)
        self.assertEqual(lists[0]['list_type'], 'unordered')

    def test_parse_ordered_list(self):
        """Test ordered list parsing."""
        text = "1. First\n2. Second\n3. Third"
        result = self.parser.parse(text)

        lists = [b for b in result['blocks'] if b['type'] == 'list']
        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0]['list_type'], 'ordered')

    def test_parse_question(self):
        """Test question detection."""
        text = "What is AI?"
        result = self.parser.parse(text)

        questions = [b for b in result['blocks'] if b['type'] == 'question']
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]['content'], "What is AI?")

    def test_parse_thinking_block(self):
        """Test thinking block parsing."""
        text = "<thinking>\nInternal reasoning\n</thinking>"
        result = self.parser.parse(text)

        thinking = [b for b in result['blocks'] if b['type'] == 'thinking']
        self.assertEqual(len(thinking), 1)
        self.assertTrue(thinking[0].get('removable'))

    def test_statistics(self):
        """Test statistics computation."""
        text = "# Title\n\nParagraph\n\n```python\ncode```"
        result = self.parser.parse(text)

        stats = result['statistics']
        self.assertGreater(stats['total_blocks'], 0)
        self.assertTrue(stats['has_code'])
        self.assertTrue(stats['has_headings'])


class TestFidelityLevels(unittest.TestCase):
    """Test fidelity level filtering."""

    def test_maximum_fidelity(self):
        """Test maximum fidelity keeps everything."""
        text = "# Title\n\n<thinking>Reasoning</thinking>\n\nContent"
        parser = StructuredParser(fidelity=FidelityLevel.MAXIMUM)
        result = parser.parse(text)

        thinking = [b for b in result['blocks'] if b['type'] == 'thinking']
        self.assertEqual(len(thinking), 1)

    def test_low_fidelity_removes_thinking(self):
        """Test low fidelity removes thinking blocks."""
        text = "# Title\n\n<thinking>Reasoning</thinking>\n\nContent"
        parser = StructuredParser(fidelity=FidelityLevel.LOW)
        result = parser.parse(text)

        thinking = [b for b in result['blocks'] if b['type'] == 'thinking']
        self.assertEqual(len(thinking), 0)


class TestRoundTrip(unittest.TestCase):
    """Test round-trip conversion."""

    def test_lossless_round_trip(self):
        """Test lossless round-trip for simple content."""
        original = "# Title\n\nParagraph content\n\n```python\ncode```"

        structured = parse_to_json(original)
        reconstructed = json_to_text(structured)

        # Should preserve structure
        self.assertIn("# Title", reconstructed)
        self.assertIn("Paragraph content", reconstructed)
        self.assertIn("```python", reconstructed)

    def test_round_trip_with_lists(self):
        """Test round-trip with lists."""
        original = "- Item 1\n- Item 2"
        structured = parse_to_json(original)
        reconstructed = json_to_text(structured)

        self.assertIn("- Item 1", reconstructed)
        self.assertIn("- Item 2", reconstructed)


class TestParseToJson(unittest.TestCase):
    """Test parse_to_json convenience function."""

    def test_parse_to_json(self):
        """Test parse_to_json function."""
        text = "# Test\n\nContent"
        result = parse_to_json(text)

        self.assertIn('version', result)
        self.assertIn('blocks', result)
        self.assertIn('statistics', result)

    def test_parse_with_fidelity(self):
        """Test parse_to_json with fidelity parameter."""
        text = "<thinking>Think</thinking>\n\nMain"
        result = parse_to_json(text, fidelity=FidelityLevel.LOW)

        thinking_blocks = [b for b in result['blocks'] if b['type'] == 'thinking']
        self.assertEqual(len(thinking_blocks), 0)


class TestJsonToText(unittest.TestCase):
    """Test json_to_text convenience function."""

    def test_json_to_text(self):
        """Test json_to_text function."""
        structured = {
            'version': '1.0',
            'fidelity': FidelityLevel.MAXIMUM,
            'blocks': [
                {'type': 'heading', 'level': 1, 'content': 'Title'},
                {'type': 'paragraph', 'content': 'Paragraph'}
            ]
        }

        text = json_to_text(structured)
        self.assertIn("# Title", text)
        self.assertIn("Paragraph", text)


if __name__ == '__main__':
    unittest.main()
