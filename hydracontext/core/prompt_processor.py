"""
Prompt Processing Module for HydraContext

Handles normalization, deduplication, and segmentation of prompts before they
reach the LLM, providing deterministic control over context window usage and
enabling memory caching for similar prompts.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
from datetime import datetime

from .text_utils import (
    segment_text,
    hash_chunk,
    is_duplicate,
    normalize_whitespace,
    extract_code_blocks,
    count_tokens_estimate
)
from ..utils.logger import get_logger
from .exceptions import ValidationError, NormalizationError, SegmentationError

logger = get_logger(__name__)


def normalize_prompt(prompt: str) -> str:
    """
    Normalize prompt formatting for consistent processing.

    Handles:
    - Line ending harmonization
    - Excess whitespace removal
    - Markdown code fence normalization
    - Multiple blank line collapsing

    Args:
        prompt: Raw prompt text

    Returns:
        Normalized prompt string

    Raises:
        ValidationError: If prompt is not a string
        NormalizationError: If normalization fails
    """
    if not isinstance(prompt, str):
        logger.error(f"Invalid prompt type: {type(prompt)}")
        raise ValidationError(f"Prompt must be a string, got {type(prompt)}")

    if not prompt:
        logger.debug("Empty prompt provided, returning empty string")
        return ""

    try:
        logger.debug(f"Normalizing prompt of length {len(prompt)}")

        # Normalize line endings
        prompt = prompt.replace('\r\n', '\n').replace('\r', '\n')

        # Strip leading/trailing whitespace
        prompt = prompt.strip()

        # Normalize code fence formatting
        # Fix: ```python\n\n\ncode -> ```python\ncode
        prompt = re.sub(r'```(\w+)?\n+', r'```\1\n', prompt)

        # Normalize closing fences
        prompt = re.sub(r'\n+```', r'\n```', prompt)

        # Collapse multiple blank lines (max 2)
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)

        # Normalize list formatting
        prompt = re.sub(r'\n([*-])\s+', r'\n\1 ', prompt)

        # Normalize heading formatting
        prompt = re.sub(r'\n(#{1,6})\s+', r'\n\1 ', prompt)

        logger.debug(f"Normalization complete, result length: {len(prompt)}")
        return prompt

    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise NormalizationError(f"Failed to normalize prompt: {e}") from e


def split_prompt(prompt: str, max_chars: int = 2048, overlap: int = 200) -> List[Dict]:
    """
    Split prompt into logical segments for processing.

    Reuses segment_text to create smaller, hashable units.

    Args:
        prompt: Normalized prompt text
        max_chars: Maximum characters per segment
        overlap: Character overlap between segments

    Returns:
        List of prompt segments with metadata
    """
    return segment_text(prompt, max_chars=max_chars, overlap=overlap)


def detect_prompt_type(text: str) -> str:
    """
    Classify prompt by intent using heuristics.

    Categories:
    - code: Contains code blocks or function definitions
    - example: Contains examples or demonstrations
    - conversation: Multi-turn dialogue format
    - instruction: Direct commands or questions
    - system: System-level directives or metadata

    Args:
        text: Prompt text to classify

    Returns:
        Detected prompt type
    """
    text_lower = text.lower()
    text_stripped = text.strip()

    # Check for system directives
    if re.search(r'^###?\s*system:', text_stripped, re.IGNORECASE | re.MULTILINE):
        return "system"

    # Check for code patterns
    if text_stripped.startswith("def ") or text_stripped.startswith("class "):
        return "code"

    if '```' in text or re.search(r'^\s{4,}\w+', text, re.MULTILINE):
        return "code"

    # Check for conversation format
    if re.search(r'(user:|assistant:|human:|ai:)', text_lower):
        return "conversation"

    # Check for examples
    if re.search(r'\b(example|e\.g\.|for instance|such as):', text_lower):
        return "example"

    if re.search(r'^(input|output):', text_lower, re.MULTILINE):
        return "example"

    # Check for instructions (questions, commands)
    if re.search(r'^(explain|describe|write|create|implement|fix|debug|how|what|why)', text_lower):
        return "instruction"

    # Default to instruction
    return "instruction"


def deduplicate_prompts(
    segments: List[Dict],
    seen_hashes: Optional[Set[str]] = None
) -> tuple[List[Dict], Set[str]]:
    """
    Mark duplicate prompt segments using hash-based deduplication.

    Returns all segments with 'duplicate' flag set appropriately.

    Args:
        segments: List of prompt segments
        seen_hashes: Set of previously seen hashes (optional)

    Returns:
        Tuple of (all_segments_with_duplicate_flags, updated_hash_set)
    """
    if seen_hashes is None:
        seen_hashes = set()

    all_segments = []

    for segment in segments:
        content = segment.get("content", "")
        content_hash = hash_chunk(content)

        if not is_duplicate(content_hash, seen_hashes):
            segment["hash"] = content_hash
            segment["duplicate"] = False
            all_segments.append(segment)
            seen_hashes.add(content_hash)
        else:
            # Still record it but mark as duplicate
            segment["hash"] = content_hash
            segment["duplicate"] = True
            all_segments.append(segment)

    return all_segments, seen_hashes


class PromptProcessor:
    """
    Main prompt processing engine.

    Provides stateful processing with caching and deduplication across
    multiple prompts.
    """

    def __init__(self, max_chars: int = 2048, overlap: int = 200):
        """
        Initialize processor.

        Args:
            max_chars: Maximum characters per segment
            overlap: Overlap between segments
        """
        self.max_chars = max_chars
        self.overlap = overlap
        self.seen_hashes: Set[str] = set()
        self.processed_count = 0

    def process(self, prompt: str, prompt_id: Optional[str] = None) -> List[Dict]:
        """
        Process a single prompt through the full pipeline.

        Args:
            prompt: Raw prompt text
            prompt_id: Optional identifier for the prompt

        Returns:
            List of processed prompt segments with metadata
        """
        if not prompt:
            return []

        # Generate ID if not provided
        if prompt_id is None:
            prompt_id = f"prompt_{self.processed_count}"

        # Step 1: Normalize
        normalized = normalize_prompt(prompt)

        # Step 2: Detect type
        prompt_type = detect_prompt_type(normalized)

        # Step 3: Split into segments
        segments = split_prompt(normalized, self.max_chars, self.overlap)

        # Step 4: Deduplicate (marks duplicates, but returns all segments)
        processed_segments, self.seen_hashes = deduplicate_prompts(
            segments,
            self.seen_hashes
        )

        # Step 5: Enrich with metadata
        enriched_segments = []
        for i, segment in enumerate(processed_segments):
            enriched = {
                "id": f"{prompt_id}_seg_{i}",
                "prompt_id": prompt_id,
                "segment_index": i,
                "type": prompt_type,
                "normalized": True,
                "content": segment["content"],
                "length": segment["length"],
                "hash": segment["hash"],
                "duplicate": segment.get("duplicate", False),
                "token_estimate": count_tokens_estimate(segment["content"]),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Extract code blocks if present
            if '```' in segment["content"]:
                enriched["code_blocks"] = extract_code_blocks(segment["content"])

            enriched_segments.append(enriched)

        self.processed_count += 1
        return enriched_segments

    def process_batch(
        self,
        prompts: List[Union[str, Dict[str, str]]]
    ) -> List[List[Dict]]:
        """
        Process multiple prompts.

        Args:
            prompts: List of prompt strings or dicts with 'id' and 'content'

        Returns:
            List of processed segment lists
        """
        results = []

        for item in prompts:
            if isinstance(item, str):
                segments = self.process(item)
            else:
                prompt_id = item.get("id")
                content = item.get("content", "")
                segments = self.process(content, prompt_id)

            results.append(segments)

        return results

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "processed_count": self.processed_count,
            "unique_hashes": len(self.seen_hashes),
            "max_chars": self.max_chars,
            "overlap": self.overlap
        }

    def reset(self):
        """Reset processor state."""
        self.seen_hashes.clear()
        self.processed_count = 0


def process_prompts(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_chars: int = 2048,
    overlap: int = 200,
    format: str = "json"
) -> Dict:
    """
    Process prompts from file and save results.

    Supports multiple input formats:
    - Plain text file (one prompt per file or separated by '---')
    - JSON file with list of prompts
    - JSONL file (one prompt per line)

    Args:
        input_path: Path to input file
        output_path: Path to output file
        max_chars: Maximum characters per segment
        overlap: Overlap between segments
        format: Output format (json, jsonl)

    Returns:
        Processing statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse based on file extension
    prompts = []

    if input_path.suffix == '.json':
        data = json.loads(content)
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and 'prompts' in data:
            prompts = data['prompts']
        else:
            prompts = [data]

    elif input_path.suffix == '.jsonl':
        prompts = [json.loads(line) for line in content.strip().split('\n')]

    else:
        # Plain text - split by '---' or treat as single prompt
        if '\n---\n' in content:
            prompts = [
                {"content": p.strip()}
                for p in content.split('\n---\n')
                if p.strip()
            ]
        else:
            prompts = [{"content": content}]

    # Process
    processor = PromptProcessor(max_chars=max_chars, overlap=overlap)
    results = processor.process_batch(prompts)

    # Flatten results
    all_segments = []
    for segments in results:
        all_segments.extend(segments)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in all_segments:
                f.write(json.dumps(segment) + '\n')
    else:
        output_data = {
            "metadata": {
                "input_file": str(input_path),
                "processed_at": datetime.utcnow().isoformat(),
                "statistics": processor.get_statistics(),
                "total_segments": len(all_segments)
            },
            "segments": all_segments
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    return processor.get_statistics()
