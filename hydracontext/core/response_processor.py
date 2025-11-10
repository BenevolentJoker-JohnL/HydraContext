"""
Response Processing Module for HydraContext

Normalizes outputs from different LLM providers, handling model-specific quirks,
formatting inconsistencies, and providing unified response structures.

This enables:
- Consistent output format across OpenAI, Anthropic, Ollama, etc.
- Artifact removal (thinking tags, system messages, etc.)
- Response comparison across models
- Streaming response normalization (especially for Ollama)
- Multi-model output aggregation
"""

import re
import json
from typing import List, Dict, Optional, Union, Iterator
from datetime import datetime

from .text_utils import (
    hash_chunk,
    is_duplicate,
    normalize_whitespace,
    extract_code_blocks,
    count_tokens_estimate
)
from .provider_parsers import UnifiedResponseParser


class ResponseNormalizer:
    """
    Normalize responses from different LLM providers into a consistent format.

    Handles provider-specific quirks and formats responses uniformly.
    """

    # Common artifacts to remove from responses
    THINKING_PATTERNS = [
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        r'\[THINKING\].*?\[/THINKING\]',
        r'Let me think.*?(?=\n\n)',
    ]

    SYSTEM_PATTERNS = [
        r'<system>.*?</system>',
        r'\[SYSTEM\].*?\[/SYSTEM\]',
        r'System:.*?(?=\n\n)',
    ]

    # Model-specific quirks
    MODEL_QUIRKS = {
        'llama2': {
            'prefix_pattern': r'^\s*\[INST\].*?\[/INST\]\s*',
            'suffix_pattern': r'\s*</s>\s*$',
        },
        'mistral': {
            'prefix_pattern': r'^\s*<\|im_start\|>.*?<\|im_end\|>\s*',
        },
        'codellama': {
            'artifact_pattern': r'```output\n.*?```',  # Often adds output blocks
        },
    }

    def __init__(
        self,
        remove_thinking: bool = True,
        remove_system: bool = True,
        normalize_code_blocks: bool = True,
        strip_artifacts: bool = True
    ):
        """
        Initialize response normalizer.

        Args:
            remove_thinking: Remove thinking/reasoning blocks
            remove_system: Remove system message artifacts
            normalize_code_blocks: Normalize code block formatting
            strip_artifacts: Remove model-specific artifacts
        """
        self.remove_thinking = remove_thinking
        self.remove_system = remove_system
        self.normalize_code_blocks = normalize_code_blocks
        self.strip_artifacts = strip_artifacts
        self.unified_parser = UnifiedResponseParser()

    def normalize(
        self,
        response: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict:
        """
        Normalize a response from any LLM provider.

        Args:
            response: Raw response text
            provider: Provider name (ollama, openai, anthropic, etc.)
            model: Specific model name

        Returns:
            Normalized response dictionary
        """
        original_response = response
        original_length = len(response)

        # Step 1: Remove thinking blocks if requested
        if self.remove_thinking:
            for pattern in self.THINKING_PATTERNS:
                response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)

        # Step 2: Remove system artifacts
        if self.remove_system:
            for pattern in self.SYSTEM_PATTERNS:
                response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)

        # Step 3: Apply model-specific cleaning
        if model and self.strip_artifacts:
            response = self._apply_model_quirks(response, model)

        # Step 4: Normalize whitespace and formatting
        response = normalize_whitespace(response)

        # Step 5: Normalize code blocks
        if self.normalize_code_blocks:
            response = self._normalize_code_blocks(response)

        # Step 6: Extract metadata
        metadata = self._extract_metadata(response)

        return {
            'normalized_content': response,
            'original_content': original_response,
            'provider': provider,
            'model': model,
            'original_length': original_length,
            'normalized_length': len(response),
            'reduction_ratio': 1 - (len(response) / original_length) if original_length > 0 else 0,
            'token_estimate': count_tokens_estimate(response),
            'metadata': metadata,
            'hash': hash_chunk(response),
            'timestamp': datetime.utcnow().isoformat()
        }

    def normalize_json_response(
        self,
        raw_response: Union[Dict, str],
        provider: Optional[str] = None
    ) -> Dict:
        """
        Normalize provider JSON response to unified format.

        This is the main method for normalizing LLM API responses.
        It handles different provider formats and returns a consistent structure.

        Args:
            raw_response: Raw response from provider (dict or JSON string)
            provider: Provider name (optional, will auto-detect)

        Returns:
            Unified response format with normalized content

        Example:
            >>> # OpenAI response
            >>> openai_resp = {"choices": [{"message": {"content": "Hello"}}], ...}
            >>> normalized = normalizer.normalize_json_response(openai_resp, "openai")
            >>>
            >>> # Ollama response
            >>> ollama_resp = {"response": "Hello", "model": "llama2", ...}
            >>> normalized = normalizer.normalize_json_response(ollama_resp, "ollama")
            >>>
            >>> # Both return same format:
            >>> print(normalized['content'])  # "Hello"
            >>> print(normalized['provider'])  # "openai" or "ollama"
        """
        # Parse JSON string if needed
        if isinstance(raw_response, str):
            try:
                raw_response = json.loads(raw_response)
            except json.JSONDecodeError:
                # Treat as plain text response
                return self.normalize(raw_response, provider=provider)

        # Use unified parser to get standard format
        parsed = self.unified_parser.parse(raw_response, provider=provider)

        # Apply content normalization to the text content
        if parsed.get('content'):
            content_normalized = self.normalize(
                parsed['content'],
                provider=parsed.get('provider'),
                model=parsed.get('model')
            )

            # Merge parsed metadata with normalized content
            return {
                **parsed,  # Keep provider-specific structure
                'content': content_normalized['normalized_content'],  # Use cleaned content
                'content_metadata': content_normalized['metadata'],  # Add text metadata
                'hash': content_normalized['hash'],  # Hash of normalized content
            }

        return parsed

    def _apply_model_quirks(self, text: str, model: str) -> str:
        """Apply model-specific cleaning patterns."""
        # Check for known model families
        for model_family, quirks in self.MODEL_QUIRKS.items():
            if model_family in model.lower():
                for pattern_name, pattern in quirks.items():
                    text = re.sub(pattern, '', text, flags=re.DOTALL)

        return text

    def _normalize_code_blocks(self, text: str) -> str:
        """Normalize code block formatting."""
        # Fix inconsistent code fences
        text = re.sub(r'```(\w+)?\s*\n+', r'```\1\n', text)
        text = re.sub(r'\n+```', r'\n```', text)

        # Ensure language tags are lowercase
        def lowercase_lang(match):
            lang = match.group(1)
            return f'```{lang.lower()}\n' if lang else '```\n'

        text = re.sub(r'```([A-Z]+)\n', lowercase_lang, text)

        return text

    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from normalized response."""
        return {
            'has_code': '```' in text,
            'code_blocks': len(extract_code_blocks(text)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'has_headings': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }


class StreamingNormalizer:
    """
    Normalize streaming responses chunk-by-chunk.

    Particularly useful for Ollama's streaming API.
    """

    def __init__(self, normalizer: Optional[ResponseNormalizer] = None):
        """
        Initialize streaming normalizer.

        Args:
            normalizer: ResponseNormalizer instance to use
        """
        self.normalizer = normalizer or ResponseNormalizer()
        self.buffer = ""
        self.chunk_count = 0

    def process_chunk(self, chunk: str) -> Optional[str]:
        """
        Process a streaming chunk.

        Args:
            chunk: New chunk of text from stream

        Returns:
            Normalized chunk if ready, None if buffering
        """
        self.buffer += chunk
        self.chunk_count += 1

        # Look for complete sentences or blocks
        if self._has_complete_unit():
            normalized = self.normalizer.normalize(self.buffer)
            output = normalized['normalized_content']
            self.buffer = ""
            return output

        return None

    def _has_complete_unit(self) -> bool:
        """Check if buffer has a complete unit (sentence, block, etc.)."""
        # Complete if ends with sentence boundary
        if re.search(r'[.!?]\s*$', self.buffer):
            return True

        # Complete if has complete code block
        if self.buffer.count('```') >= 2 and self.buffer.count('```') % 2 == 0:
            return True

        # Complete if has paragraph break
        if '\n\n' in self.buffer:
            return True

        return False

    def flush(self) -> Optional[str]:
        """Flush remaining buffer."""
        if self.buffer:
            normalized = self.normalizer.normalize(self.buffer)
            output = normalized['normalized_content']
            self.buffer = ""
            return output
        return None


class ResponseComparator:
    """
    Compare responses from different models/providers.

    Useful for multi-model workflows and quality assessment.
    """

    def __init__(self):
        """Initialize comparator."""
        self.seen_hashes = set()

    def compare(
        self,
        responses: List[Dict],
        similarity_threshold: float = 0.8
    ) -> Dict:
        """
        Compare multiple normalized responses.

        Args:
            responses: List of normalized response dicts
            similarity_threshold: Threshold for considering responses similar

        Returns:
            Comparison analysis
        """
        if not responses:
            return {'error': 'No responses provided'}

        # Compute pairwise similarities
        similarities = []
        for i, resp1 in enumerate(responses):
            for j, resp2 in enumerate(responses[i+1:], i+1):
                sim = self._compute_similarity(
                    resp1['normalized_content'],
                    resp2['normalized_content']
                )
                similarities.append({
                    'pair': (i, j),
                    'models': (resp1.get('model'), resp2.get('model')),
                    'similarity': sim,
                    'similar': sim >= similarity_threshold
                })

        # Find duplicates
        duplicates = []
        for resp in responses:
            content_hash = resp['hash']
            is_dup = is_duplicate(content_hash, self.seen_hashes)
            duplicates.append(is_dup)
            self.seen_hashes.add(content_hash)

        # Aggregate statistics
        return {
            'total_responses': len(responses),
            'unique_responses': sum(1 for d in duplicates if not d),
            'duplicate_responses': sum(1 for d in duplicates if d),
            'pairwise_similarities': similarities,
            'average_similarity': sum(s['similarity'] for s in similarities) / len(similarities) if similarities else 0,
            'providers': list(set(r.get('provider') for r in responses if r.get('provider'))),
            'models': list(set(r.get('model') for r in responses if r.get('model'))),
            'length_stats': {
                'min': min(r['normalized_length'] for r in responses),
                'max': max(r['normalized_length'] for r in responses),
                'avg': sum(r['normalized_length'] for r in responses) / len(responses),
            },
            'token_stats': {
                'min': min(r['token_estimate'] for r in responses),
                'max': max(r['token_estimate'] for r in responses),
                'avg': sum(r['token_estimate'] for r in responses) / len(responses),
            }
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using simple overlap.

        For production, could use more sophisticated methods like:
        - Embeddings + cosine similarity
        - Edit distance
        - BLEU/ROUGE scores
        """
        # Simple word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class OllamaResponseHandler:
    """
    Specialized handler for Ollama-specific features.

    Ollama provides:
    - Streaming support
    - Context window info
    - Multiple model support on same endpoint
    - Token usage stats
    """

    def __init__(self, normalizer: Optional[ResponseNormalizer] = None):
        """Initialize Ollama handler."""
        self.normalizer = normalizer or ResponseNormalizer()
        self.streaming_normalizer = StreamingNormalizer(self.normalizer)

    def normalize_ollama_response(self, response_dict: Dict) -> Dict:
        """
        Normalize Ollama API response format.

        Ollama returns: {'model': str, 'response': str, 'done': bool, ...}

        Args:
            response_dict: Raw Ollama response dict

        Returns:
            Normalized response with Ollama-specific metadata
        """
        content = response_dict.get('response', response_dict.get('message', {}).get('content', ''))
        model = response_dict.get('model', 'unknown')

        normalized = self.normalizer.normalize(
            content,
            provider='ollama',
            model=model
        )

        # Add Ollama-specific metadata
        normalized['ollama_metadata'] = {
            'model': model,
            'done': response_dict.get('done', True),
            'total_duration': response_dict.get('total_duration'),
            'load_duration': response_dict.get('load_duration'),
            'prompt_eval_count': response_dict.get('prompt_eval_count'),
            'eval_count': response_dict.get('eval_count'),
            'context': response_dict.get('context'),  # For context window tracking
        }

        return normalized

    def process_streaming_response(self, stream: Iterator[Dict]) -> Iterator[Dict]:
        """
        Process Ollama streaming response.

        Args:
            stream: Iterator of Ollama response chunks

        Yields:
            Normalized chunks
        """
        for chunk in stream:
            content = chunk.get('response', '')

            # Process through streaming normalizer
            normalized_chunk = self.streaming_normalizer.process_chunk(content)

            if normalized_chunk:
                yield {
                    'content': normalized_chunk,
                    'model': chunk.get('model'),
                    'done': chunk.get('done', False),
                    'provider': 'ollama'
                }

        # Flush remaining buffer
        final = self.streaming_normalizer.flush()
        if final:
            yield {
                'content': final,
                'done': True,
                'provider': 'ollama'
            }
