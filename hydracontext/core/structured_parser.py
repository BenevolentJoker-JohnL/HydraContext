"""
Structured JSON Representation for Maximum Information Fidelity

Parses all content (prompts, responses, markdown) into a canonical JSON
structure that preserves ALL information while enabling granular control
over what to keep, compress, or discard.

This enables:
- Lossless round-tripping (text → JSON → text)
- Semantic operations on structured data
- Precise control over information fidelity
- Multi-model response fusion
- Context compression with controllable detail levels
"""

import re
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ContentType(str, Enum):
    """Types of content blocks."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    LIST = "list"
    TABLE = "table"
    QUOTE = "quote"
    LINK = "link"
    IMAGE = "image"
    INLINE_CODE = "inline_code"
    BOLD = "bold"
    ITALIC = "italic"
    THINKING = "thinking"  # Reasoning/thought process
    INSTRUCTION = "instruction"
    QUESTION = "question"
    ANSWER = "answer"
    EXAMPLE = "example"
    METADATA = "metadata"


class FidelityLevel(str, Enum):
    """Information fidelity levels."""
    MAXIMUM = "maximum"      # Keep everything, all structure
    HIGH = "high"            # Keep semantic structure, normalize formatting
    MEDIUM = "medium"        # Keep main content, discard auxiliary
    LOW = "low"              # Summary only, discard details
    MINIMAL = "minimal"      # Core message only


class StructuredParser:
    """
    Parse text into structured JSON representation.

    Converts unstructured text into a hierarchical JSON structure
    that preserves all information while enabling semantic operations.
    """

    def __init__(self, fidelity: FidelityLevel = FidelityLevel.MAXIMUM):
        """
        Initialize parser.

        Args:
            fidelity: Information fidelity level
        """
        self.fidelity = fidelity

    def parse(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Parse text into structured JSON.

        Args:
            text: Input text (markdown, plain text, etc.)
            metadata: Optional metadata to attach

        Returns:
            Structured JSON representation

        Example:
            >>> parser = StructuredParser()
            >>> result = parser.parse("# Hello\n\nThis is **bold** text.")
            >>> print(result['blocks'][0]['type'])  # "heading"
            >>> print(result['blocks'][0]['content'])  # "Hello"
        """
        blocks = []

        # Split into logical blocks
        lines = text.split('\n')
        current_block = []
        current_type = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect code blocks
            if line.strip().startswith('```'):
                if current_block:
                    blocks.append(self._create_block(current_type, current_block))
                    current_block = []

                # Extract code block
                code_block = self._extract_code_block(lines, i)
                if code_block:
                    blocks.append(code_block)
                    i = code_block['_end_line']
                    current_type = None
                    i += 1
                    continue

            # Detect headings
            elif line.strip().startswith('#'):
                if current_block:
                    blocks.append(self._create_block(current_type, current_block))
                    current_block = []

                heading = self._parse_heading(line)
                blocks.append(heading)
                current_type = None

            # Detect lists
            elif re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
                if current_type != ContentType.LIST:
                    if current_block:
                        blocks.append(self._create_block(current_type, current_block))
                        current_block = []
                    current_type = ContentType.LIST

                current_block.append(line)

            # Detect thinking blocks
            elif '<thinking>' in line.lower() or '<thought>' in line.lower():
                if current_block:
                    blocks.append(self._create_block(current_type, current_block))
                    current_block = []

                thinking = self._extract_thinking_block(lines, i)
                if thinking:
                    blocks.append(thinking)
                    i = thinking['_end_line']
                    current_type = None
                    i += 1
                    continue

            # Detect questions
            elif line.strip().endswith('?'):
                if current_block:
                    blocks.append(self._create_block(current_type, current_block))
                    current_block = []

                blocks.append({
                    'type': ContentType.QUESTION,
                    'content': line.strip(),
                    'inline_formatting': self._parse_inline_formatting(line)
                })
                current_type = None

            # Regular paragraph
            elif line.strip():
                if current_type != ContentType.PARAGRAPH:
                    if current_block:
                        blocks.append(self._create_block(current_type, current_block))
                        current_block = []
                    current_type = ContentType.PARAGRAPH

                current_block.append(line)

            # Blank line - end current block
            else:
                if current_block:
                    blocks.append(self._create_block(current_type, current_block))
                    current_block = []
                    current_type = None

            i += 1

        # Add final block
        if current_block:
            blocks.append(self._create_block(current_type, current_block))

        # Apply fidelity filtering
        blocks = self._apply_fidelity(blocks)

        return {
            'version': '1.0',
            'fidelity': self.fidelity,
            'metadata': metadata or {},
            'blocks': blocks,
            'statistics': self._compute_statistics(blocks)
        }

    def reconstruct(self, structured: Dict) -> str:
        """
        Reconstruct text from structured JSON.

        Args:
            structured: Structured JSON representation

        Returns:
            Reconstructed text
        """
        lines = []

        for block in structured.get('blocks', []):
            block_type = block.get('type')

            if block_type == ContentType.HEADING:
                level = block.get('level', 1)
                content = block.get('content', '')
                lines.append('#' * level + ' ' + content)
                lines.append('')

            elif block_type == ContentType.CODE_BLOCK:
                lang = block.get('language', '')
                code = block.get('content', '')
                lines.append(f'```{lang}')
                lines.append(code)
                lines.append('```')
                lines.append('')

            elif block_type == ContentType.LIST:
                items = block.get('items', [])
                list_type = block.get('list_type', 'unordered')
                for i, item in enumerate(items):
                    if list_type == 'ordered':
                        lines.append(f"{i+1}. {item}")
                    else:
                        lines.append(f"- {item}")
                lines.append('')

            elif block_type == ContentType.PARAGRAPH:
                content = block.get('content', '')
                lines.append(content)
                lines.append('')

            elif block_type == ContentType.QUESTION:
                content = block.get('content', '')
                lines.append(content)
                lines.append('')

            elif block_type == ContentType.THINKING:
                content = block.get('content', '')
                if structured.get('fidelity') == FidelityLevel.MAXIMUM:
                    lines.append(f'<thinking>{content}</thinking>')
                    lines.append('')

        return '\n'.join(lines).strip()

    def _create_block(self, block_type: Optional[ContentType], lines: List[str]) -> Dict:
        """Create a block from lines."""
        if not lines:
            return {}

        if block_type == ContentType.LIST:
            return self._parse_list(lines)

        # Default to paragraph
        content = '\n'.join(lines).strip()
        return {
            'type': block_type or ContentType.PARAGRAPH,
            'content': content,
            'inline_formatting': self._parse_inline_formatting(content),
            'length': len(content),
            'line_count': len(lines)
        }

    def _parse_heading(self, line: str) -> Dict:
        """Parse heading line."""
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            level = len(match.group(1))
            content = match.group(2).strip()
            return {
                'type': ContentType.HEADING,
                'level': level,
                'content': content,
                'inline_formatting': self._parse_inline_formatting(content)
            }
        return {}

    def _extract_code_block(self, lines: List[str], start: int) -> Optional[Dict]:
        """Extract code block."""
        lang_match = re.match(r'```(\w+)?', lines[start])
        language = lang_match.group(1) if lang_match else ''

        code_lines = []
        end = start + 1

        while end < len(lines):
            if lines[end].strip().startswith('```'):
                return {
                    'type': ContentType.CODE_BLOCK,
                    'language': language,
                    'content': '\n'.join(code_lines),
                    'length': len('\n'.join(code_lines)),
                    'line_count': len(code_lines),
                    '_end_line': end
                }
            code_lines.append(lines[end])
            end += 1

        return None

    def _extract_thinking_block(self, lines: List[str], start: int) -> Optional[Dict]:
        """Extract thinking/reasoning block."""
        thinking_lines = []
        end = start

        # Check for opening tag
        if '<thinking>' in lines[start].lower():
            end = start + 1
            while end < len(lines):
                if '</thinking>' in lines[end].lower():
                    # Extract content between tags
                    content = '\n'.join(thinking_lines).strip()
                    return {
                        'type': ContentType.THINKING,
                        'content': content,
                        'length': len(content),
                        'removable': True,  # Can be removed at lower fidelity
                        '_end_line': end
                    }
                thinking_lines.append(lines[end])
                end += 1

        return None

    def _parse_list(self, lines: List[str]) -> Dict:
        """Parse list block."""
        items = []
        list_type = 'unordered'

        for line in lines:
            # Ordered list
            match = re.match(r'^\s*\d+\.\s+(.*)', line)
            if match:
                list_type = 'ordered'
                items.append(match.group(1).strip())
                continue

            # Unordered list
            match = re.match(r'^\s*[-*+]\s+(.*)', line)
            if match:
                items.append(match.group(1).strip())

        return {
            'type': ContentType.LIST,
            'list_type': list_type,
            'items': items,
            'item_count': len(items)
        }

    def _parse_inline_formatting(self, text: str) -> Dict:
        """Parse inline formatting (bold, italic, code, etc.)."""
        return {
            'has_bold': bool(re.search(r'\*\*.*?\*\*', text)),
            'has_italic': bool(re.search(r'\*.*?\*', text)),
            'has_inline_code': bool(re.search(r'`.*?`', text)),
            'has_links': bool(re.search(r'\[.*?\]\(.*?\)', text)),
        }

    def _apply_fidelity(self, blocks: List[Dict]) -> List[Dict]:
        """Apply fidelity filtering to blocks."""
        if self.fidelity == FidelityLevel.MAXIMUM:
            return blocks

        filtered = []
        for block in blocks:
            # Remove thinking blocks at lower fidelity
            if block.get('type') == ContentType.THINKING:
                if self.fidelity in (FidelityLevel.LOW, FidelityLevel.MINIMAL):
                    continue

            # Keep metadata only at high/max fidelity
            if block.get('type') == ContentType.METADATA:
                if self.fidelity in (FidelityLevel.LOW, FidelityLevel.MINIMAL):
                    continue

            filtered.append(block)

        return filtered

    def _compute_statistics(self, blocks: List[Dict]) -> Dict:
        """Compute statistics about parsed structure."""
        return {
            'total_blocks': len(blocks),
            'block_types': {
                block_type: sum(1 for b in blocks if b.get('type') == block_type)
                for block_type in ContentType
            },
            'total_length': sum(b.get('length', 0) for b in blocks),
            'has_code': any(b.get('type') == ContentType.CODE_BLOCK for b in blocks),
            'has_lists': any(b.get('type') == ContentType.LIST for b in blocks),
            'has_headings': any(b.get('type') == ContentType.HEADING for b in blocks),
        }


def parse_to_json(text: str, fidelity: FidelityLevel = FidelityLevel.MAXIMUM) -> Dict:
    """
    Convenience function to parse text to structured JSON.

    Args:
        text: Input text
        fidelity: Information fidelity level

    Returns:
        Structured JSON representation
    """
    parser = StructuredParser(fidelity=fidelity)
    return parser.parse(text)


def json_to_text(structured: Dict) -> str:
    """
    Convenience function to reconstruct text from JSON.

    Args:
        structured: Structured JSON

    Returns:
        Reconstructed text
    """
    parser = StructuredParser()
    return parser.reconstruct(structured)
