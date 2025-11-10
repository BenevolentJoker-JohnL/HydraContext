"""
Core text processing utilities for HydraContext.

Provides fundamental operations for text segmentation, hashing, and deduplication
that are used across both repository processing and prompt processing.
"""

import hashlib
import re
from typing import List, Dict, Set, Optional


def segment_text(text: str, max_chars: int = 2048, overlap: int = 200) -> List[Dict[str, any]]:
    """
    Segment text into chunks with optional overlap for context preservation.

    Args:
        text: Input text to segment
        max_chars: Maximum characters per segment
        overlap: Number of characters to overlap between segments

    Returns:
        List of segment dictionaries with metadata
    """
    if not text or max_chars <= 0:
        return []

    segments = []
    lines = text.split('\n')
    current_segment = []
    current_length = 0
    segment_id = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for newline

        # If single line exceeds max_chars, split it
        if line_length > max_chars:
            if current_segment:
                segments.append({
                    "id": segment_id,
                    "content": '\n'.join(current_segment),
                    "length": current_length,
                    "type": "segment"
                })
                segment_id += 1
                current_segment = []
                current_length = 0

            # Split long line into chunks
            for i in range(0, len(line), max_chars - overlap):
                chunk = line[i:i + max_chars]
                segments.append({
                    "id": segment_id,
                    "content": chunk,
                    "length": len(chunk),
                    "type": "long_line_fragment"
                })
                segment_id += 1
            continue

        # Check if adding line would exceed limit
        if current_length + line_length > max_chars and current_segment:
            # Save current segment
            segments.append({
                "id": segment_id,
                "content": '\n'.join(current_segment),
                "length": current_length,
                "type": "segment"
            })
            segment_id += 1

            # Start new segment with overlap
            if overlap > 0 and current_segment:
                overlap_text = '\n'.join(current_segment)
                overlap_lines = []
                overlap_length = 0

                # Take last few lines for overlap
                for prev_line in reversed(current_segment):
                    if overlap_length + len(prev_line) + 1 <= overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_length += len(prev_line) + 1
                    else:
                        break

                current_segment = overlap_lines
                current_length = overlap_length
            else:
                current_segment = []
                current_length = 0

        current_segment.append(line)
        current_length += line_length

    # Add final segment
    if current_segment:
        segments.append({
            "id": segment_id,
            "content": '\n'.join(current_segment),
            "length": current_length,
            "type": "segment"
        })

    return segments


def hash_chunk(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a cryptographic hash of text content.

    Args:
        text: Input text to hash
        algorithm: Hash algorithm (sha256, md5, sha1)

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "md5":
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    else:  # default to sha256
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


def is_duplicate(hash_value: str, seen_hashes: Set[str]) -> bool:
    """
    Check if a hash has been seen before (for deduplication).

    Args:
        hash_value: Hash to check
        seen_hashes: Set of previously seen hashes

    Returns:
        True if duplicate, False if unique
    """
    return hash_value in seen_hashes


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving intentional formatting.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]

    # Collapse multiple blank lines to max 2
    result = []
    blank_count = 0

    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)

    return '\n'.join(result).strip()


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.

    Args:
        text: Markdown text potentially containing code blocks

    Returns:
        List of code block dictionaries with language and content
    """
    # Pattern for fenced code blocks
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.finditer(pattern, text, re.DOTALL)

    blocks = []
    for match in matches:
        language = match.group(1) or "plaintext"
        content = match.group(2).strip()
        blocks.append({
            "language": language,
            "content": content,
            "start": match.start(),
            "end": match.end()
        })

    return blocks


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count (4 chars ≈ 1 token for English).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 characters per token
    # More accurate would use tiktoken, but this avoids dependency
    return len(text) // 4
