"""
Provider-Specific Response Parsers

Normalizes different provider response formats into unified JSON structure.

Standard Format:
{
    "content": str,           # The actual response text
    "provider": str,          # Provider name
    "model": str,             # Model name
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    },
    "finish_reason": str,     # why generation stopped
    "metadata": dict,         # Provider-specific metadata
    "timestamp": str          # ISO format timestamp
}
"""

from typing import Dict, Any, Optional
from datetime import datetime


class ProviderParser:
    """Base class for provider-specific parsers."""

    def parse(self, raw_response: Any) -> Dict:
        """
        Parse provider response into standard format.

        Args:
            raw_response: Raw response from provider

        Returns:
            Standardized response dict
        """
        raise NotImplementedError


class OpenAIParser(ProviderParser):
    """Parse OpenAI API responses."""

    def parse(self, raw_response: Any) -> Dict:
        """
        Parse OpenAI response format.

        OpenAI format:
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        """
        if isinstance(raw_response, dict):
            choice = raw_response.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '')

            return {
                "content": content,
                "provider": "openai",
                "model": raw_response.get('model', 'unknown'),
                "usage": raw_response.get('usage', {}),
                "finish_reason": choice.get('finish_reason', 'unknown'),
                "metadata": {
                    "id": raw_response.get('id'),
                    "created": raw_response.get('created'),
                    "object": raw_response.get('object'),
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        # Handle string response
        return {
            "content": str(raw_response),
            "provider": "openai",
            "model": "unknown",
            "usage": {},
            "finish_reason": "unknown",
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat()
        }


class AnthropicParser(ProviderParser):
    """Parse Anthropic Claude API responses."""

    def parse(self, raw_response: Any) -> Dict:
        """
        Parse Anthropic response format.

        Anthropic format:
        {
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I assist you today?"
                }
            ],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25
            }
        }
        """
        if isinstance(raw_response, dict):
            # Extract text from content array
            content_array = raw_response.get('content', [])
            if isinstance(content_array, list):
                text_parts = [
                    item.get('text', '')
                    for item in content_array
                    if item.get('type') == 'text'
                ]
                content = '\n'.join(text_parts)
            else:
                content = str(content_array)

            # Normalize usage format
            usage = raw_response.get('usage', {})
            normalized_usage = {
                "prompt_tokens": usage.get('input_tokens', 0),
                "completion_tokens": usage.get('output_tokens', 0),
                "total_tokens": usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            }

            return {
                "content": content,
                "provider": "anthropic",
                "model": raw_response.get('model', 'unknown'),
                "usage": normalized_usage,
                "finish_reason": raw_response.get('stop_reason', 'unknown'),
                "metadata": {
                    "id": raw_response.get('id'),
                    "type": raw_response.get('type'),
                    "stop_sequence": raw_response.get('stop_sequence'),
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        # Handle string response
        return {
            "content": str(raw_response),
            "provider": "anthropic",
            "model": "unknown",
            "usage": {},
            "finish_reason": "unknown",
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat()
        }


class OllamaParser(ProviderParser):
    """Parse Ollama API responses."""

    def parse(self, raw_response: Any) -> Dict:
        """
        Parse Ollama response format.

        Ollama format:
        {
            "model": "llama2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "response": "Hello! How can I assist you today?",
            "done": true,
            "context": [1, 2, 3, ...],
            "total_duration": 4883583458,
            "load_duration": 1334875,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 342546000,
            "eval_count": 282,
            "eval_duration": 4535599000
        }
        """
        if isinstance(raw_response, dict):
            # Extract content
            content = raw_response.get('response', raw_response.get('message', {}).get('content', ''))

            # Calculate token usage from eval counts
            prompt_tokens = raw_response.get('prompt_eval_count', 0)
            completion_tokens = raw_response.get('eval_count', 0)

            normalized_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

            # Determine finish reason
            finish_reason = "stop" if raw_response.get('done', False) else "length"

            return {
                "content": content,
                "provider": "ollama",
                "model": raw_response.get('model', 'unknown'),
                "usage": normalized_usage,
                "finish_reason": finish_reason,
                "metadata": {
                    "created_at": raw_response.get('created_at'),
                    "done": raw_response.get('done'),
                    "context": raw_response.get('context'),  # Important for context tracking
                    "total_duration": raw_response.get('total_duration'),
                    "load_duration": raw_response.get('load_duration'),
                    "prompt_eval_duration": raw_response.get('prompt_eval_duration'),
                    "eval_duration": raw_response.get('eval_duration'),
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        # Handle string response
        return {
            "content": str(raw_response),
            "provider": "ollama",
            "model": "unknown",
            "usage": {},
            "finish_reason": "unknown",
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat()
        }


class GenericParser(ProviderParser):
    """Parse generic/unknown provider responses."""

    def parse(self, raw_response: Any) -> Dict:
        """
        Parse generic response.

        Tries to extract content intelligently from various formats.
        """
        content = ""

        if isinstance(raw_response, str):
            content = raw_response

        elif isinstance(raw_response, dict):
            # Try common content fields
            for field in ['content', 'response', 'text', 'message', 'output']:
                if field in raw_response:
                    content = str(raw_response[field])
                    break

            # If no content found, use entire dict as string
            if not content:
                content = str(raw_response)

        else:
            content = str(raw_response)

        return {
            "content": content,
            "provider": "generic",
            "model": "unknown",
            "usage": {},
            "finish_reason": "unknown",
            "metadata": raw_response if isinstance(raw_response, dict) else {},
            "timestamp": datetime.utcnow().isoformat()
        }


class UnifiedResponseParser:
    """
    Unified parser that routes to provider-specific parsers.

    Auto-detects provider from response format and normalizes to standard JSON.
    """

    def __init__(self):
        """Initialize unified parser."""
        self.parsers = {
            'openai': OpenAIParser(),
            'anthropic': AnthropicParser(),
            'ollama': OllamaParser(),
            'generic': GenericParser(),
        }

    def parse(
        self,
        raw_response: Any,
        provider: Optional[str] = None
    ) -> Dict:
        """
        Parse response from any provider into standard format.

        Args:
            raw_response: Raw response from LLM provider
            provider: Provider name (optional, will auto-detect if not provided)

        Returns:
            Standardized response dict
        """
        # If provider specified, use that parser
        if provider and provider in self.parsers:
            return self.parsers[provider].parse(raw_response)

        # Otherwise, try to auto-detect provider
        detected_provider = self._detect_provider(raw_response)
        return self.parsers[detected_provider].parse(raw_response)

    def _detect_provider(self, raw_response: Any) -> str:
        """
        Auto-detect provider from response format.

        Args:
            raw_response: Raw response

        Returns:
            Detected provider name
        """
        if not isinstance(raw_response, dict):
            return 'generic'

        # OpenAI: has 'choices' array
        if 'choices' in raw_response and 'usage' in raw_response:
            return 'openai'

        # Anthropic: has content array with text items
        if 'content' in raw_response and isinstance(raw_response.get('content'), list):
            if any(item.get('type') == 'text' for item in raw_response['content']):
                return 'anthropic'

        # Ollama: has 'response' field and 'done' flag
        if 'response' in raw_response and 'done' in raw_response:
            return 'ollama'

        # Ollama alternative format (chat)
        if 'message' in raw_response and 'model' in raw_response:
            return 'ollama'

        return 'generic'
