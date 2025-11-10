"""
High-level API for programmatic use of HydraContext.

This module provides simple, convenient functions for integrating
HydraContext into your application.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path

from .core.prompt_processor import PromptProcessor


class HydraContextAPI:
    """
    Main API class for HydraContext.

    Provides a simple, stateful interface for processing prompts
    with automatic deduplication and caching.

    Example:
        >>> hydra = HydraContext()
        >>> result = hydra.process("Explain transformers")
        >>> print(result[0]['type'])  # "instruction"
        >>>
        >>> # Batch processing with deduplication
        >>> prompts = ["What is AI?", "Explain ML", "What is AI?"]
        >>> results = hydra.process_batch(prompts)
        >>> print(hydra.stats())  # Shows 2 unique prompts
    """

    def __init__(
        self,
        max_chars: int = 2048,
        overlap: int = 200,
        auto_deduplicate: bool = True
    ):
        """
        Initialize HydraContext.

        Args:
            max_chars: Maximum characters per segment
            overlap: Overlap between segments for context preservation
            auto_deduplicate: Automatically deduplicate across all prompts
        """
        self.processor = PromptProcessor(max_chars=max_chars, overlap=overlap)
        self.auto_deduplicate = auto_deduplicate

    def process(
        self,
        prompt: str,
        prompt_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a single prompt.

        Args:
            prompt: Raw prompt text
            prompt_id: Optional identifier

        Returns:
            List of processed segments with metadata

        Example:
            >>> hydra = HydraContext()
            >>> segments = hydra.process("Explain how GPT works")
            >>> print(segments[0]['type'])  # "instruction"
            >>> print(segments[0]['token_estimate'])  # Estimated tokens
        """
        return self.processor.process(prompt, prompt_id)

    def process_batch(
        self,
        prompts: List[Union[str, Dict[str, str]]]
    ) -> List[List[Dict]]:
        """
        Process multiple prompts with deduplication.

        Args:
            prompts: List of prompt strings or dicts with 'id' and 'content'

        Returns:
            List of segment lists

        Example:
            >>> hydra = HydraContext()
            >>> prompts = [
            ...     "What is AI?",
            ...     {"id": "ml_question", "content": "Explain ML"}
            ... ]
            >>> results = hydra.process_batch(prompts)
            >>> len(results)  # 2 (one per prompt)
        """
        return self.processor.process_batch(prompts)

    def stats(self) -> Dict:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing metrics

        Example:
            >>> hydra = HydraContext()
            >>> hydra.process("Test prompt")
            >>> stats = hydra.stats()
            >>> print(stats['processed_count'])  # 1
            >>> print(stats['unique_hashes'])  # 1
        """
        return self.processor.get_statistics()

    def reset(self):
        """
        Reset processor state (clear cache and counters).

        Example:
            >>> hydra = HydraContext()
            >>> hydra.process("Test")
            >>> hydra.reset()
            >>> hydra.stats()['processed_count']  # 0
        """
        self.processor.reset()

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"HydraContext(processed={stats['processed_count']}, "
            f"unique={stats['unique_hashes']}, "
            f"max_chars={stats['max_chars']})"
        )


# Convenience functions for one-off operations

def normalize(prompt: str) -> str:
    """
    Normalize a prompt without full processing.

    Args:
        prompt: Raw prompt text

    Returns:
        Normalized prompt string

    Example:
        >>> from hydra_context import normalize
        >>> clean = normalize("   messy   prompt   \\n\\n\\n\\n   ")
        >>> print(clean)  # "messy prompt"
    """
    from .prompt_processor import normalize_prompt
    return normalize_prompt(prompt)


def classify(prompt: str) -> str:
    """
    Classify a prompt by type.

    Args:
        prompt: Prompt text

    Returns:
        Prompt type (instruction, code, conversation, example, system)

    Example:
        >>> from hydra_context import classify
        >>> classify("```python\\ndef foo(): pass```")  # "code"
        >>> classify("Explain how LLMs work")  # "instruction"
    """
    from .prompt_processor import detect_prompt_type
    return detect_prompt_type(prompt)


def chunk(
    text: str,
    max_chars: int = 2048,
    overlap: int = 200
) -> List[str]:
    """
    Split text into chunks with overlap.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks

    Example:
        >>> from hydra_context import chunk
        >>> long_text = "..." * 10000
        >>> chunks = chunk(long_text, max_chars=1024)
        >>> len(chunks)  # Multiple chunks
    """
    from .text_utils import segment_text
    segments = segment_text(text, max_chars=max_chars, overlap=overlap)
    return [seg['content'] for seg in segments]


def quick_process(
    prompt: str,
    max_chars: int = 2048
) -> Dict:
    """
    Quick one-off processing with simplified output.

    Args:
        prompt: Prompt to process
        max_chars: Maximum characters per segment

    Returns:
        Dictionary with summary info

    Example:
        >>> from hydra_context import quick_process
        >>> result = quick_process("Explain transformers")
        >>> print(result['type'])  # "instruction"
        >>> print(result['segments'])  # Number of segments
    """
    hydra = HydraContext(max_chars=max_chars)
    segments = hydra.process(prompt)

    return {
        'type': segments[0]['type'] if segments else 'unknown',
        'segments': len(segments),
        'total_length': sum(s['length'] for s in segments),
        'token_estimate': sum(s['token_estimate'] for s in segments),
        'content': [s['content'] for s in segments]
    }


def load_and_process(
    file_path: Union[str, Path],
    max_chars: int = 2048,
    overlap: int = 200
) -> List[Dict]:
    """
    Load prompts from file and process them.

    Supports text, JSON, and JSONL formats.

    Args:
        file_path: Path to input file
        max_chars: Maximum characters per segment
        overlap: Overlap between segments

    Returns:
        List of all processed segments

    Example:
        >>> from hydra_context import load_and_process
        >>> segments = load_and_process("prompts.txt")
        >>> for seg in segments:
        ...     print(seg['type'], seg['content'][:50])
    """
    import json

    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse based on extension
    prompts = []
    if file_path.suffix == '.json':
        data = json.loads(content)
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and 'prompts' in data:
            prompts = data['prompts']
        else:
            prompts = [data]
    elif file_path.suffix == '.jsonl':
        prompts = [json.loads(line) for line in content.strip().split('\n')]
    else:
        # Plain text
        if '\n---\n' in content:
            prompts = [p.strip() for p in content.split('\n---\n') if p.strip()]
        else:
            prompts = [content]

    # Process
    hydra = HydraContext(max_chars=max_chars, overlap=overlap)
    results = hydra.process_batch(prompts)

    # Flatten
    all_segments = []
    for segments in results:
        all_segments.extend(segments)

    return all_segments
