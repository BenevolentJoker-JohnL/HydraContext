"""
Bidirectional Context Normalization for HydraContext

Provides unified interface for normalizing both:
- INPUT: Prompts going TO the LLM (preprocessing)
- OUTPUT: Responses coming FROM the LLM (postprocessing)

This enables consistent context handling across different LLM providers.
"""

from typing import Dict, Optional, List
from .prompt_processor import PromptProcessor
from .response_processor import ResponseNormalizer, OllamaResponseHandler


class ContextNormalizer:
    """
    Bidirectional context normalizer.

    Normalizes prompts before sending to LLM and responses after receiving.
    Works with any LLM provider (OpenAI, Anthropic, Ollama, etc.).

    Example:
        >>> normalizer = ContextNormalizer()
        >>>
        >>> # Normalize input
        >>> prompt_data = normalizer.normalize_input("Explain AI")
        >>> clean_prompt = prompt_data['content']
        >>>
        >>> # Send to your LLM...
        >>> response = my_llm.generate(clean_prompt)
        >>>
        >>> # Normalize output
        >>> response_data = normalizer.normalize_output(
        ...     response,
        ...     provider='ollama',
        ...     model='llama2'
        ... )
        >>> clean_response = response_data['normalized_content']
    """

    def __init__(
        self,
        max_chars: int = 2048,
        overlap: int = 200,
        remove_thinking: bool = True,
        normalize_code_blocks: bool = True
    ):
        """
        Initialize bidirectional normalizer.

        Args:
            max_chars: Max chars per prompt segment
            overlap: Overlap for prompt segmentation
            remove_thinking: Remove thinking tags from responses
            normalize_code_blocks: Normalize code block formatting
        """
        self.prompt_processor = PromptProcessor(
            max_chars=max_chars,
            overlap=overlap
        )

        self.response_normalizer = ResponseNormalizer(
            remove_thinking=remove_thinking,
            normalize_code_blocks=normalize_code_blocks
        )

        self.ollama_handler = OllamaResponseHandler(self.response_normalizer)

    def normalize_input(
        self,
        prompt: str,
        prompt_id: Optional[str] = None
    ) -> Dict:
        """
        Normalize input prompt before sending to LLM.

        Args:
            prompt: Raw user prompt
            prompt_id: Optional prompt identifier

        Returns:
            Dict with normalized prompt and metadata
        """
        segments = self.prompt_processor.process(prompt, prompt_id)

        # For single-segment prompts, return simplified format
        if len(segments) == 1:
            return {
                'content': segments[0]['content'],
                'type': segments[0]['type'],
                'token_estimate': segments[0]['token_estimate'],
                'segments': segments,
                'direction': 'input'
            }

        # For multi-segment prompts, return all segments
        return {
            'content': '\n\n'.join(s['content'] for s in segments),
            'type': segments[0]['type'],
            'token_estimate': sum(s['token_estimate'] for s in segments),
            'segments': segments,
            'direction': 'input'
        }

    def normalize_output(
        self,
        response: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict:
        """
        Normalize output response from LLM.

        Args:
            response: Raw LLM response
            provider: Provider name (ollama, openai, anthropic)
            model: Model name

        Returns:
            Dict with normalized response and metadata
        """
        normalized = self.response_normalizer.normalize(
            response,
            provider=provider,
            model=model
        )

        normalized['direction'] = 'output'
        return normalized

    def normalize_ollama_output(self, response_dict: Dict) -> Dict:
        """
        Normalize Ollama-specific response format.

        Args:
            response_dict: Raw Ollama API response

        Returns:
            Normalized response with Ollama metadata
        """
        # Use the JSON normalizer for consistent output
        normalized = self.response_normalizer.normalize_json_response(
            response_dict,
            provider='ollama'
        )
        normalized['direction'] = 'output'
        return normalized

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'input_processor': self.prompt_processor.get_statistics(),
        }


class MultiProviderNormalizer:
    """
    Normalize context across multiple LLM providers simultaneously.

    Useful for:
    - Comparing outputs from different models
    - Routing prompts to different providers
    - Aggregating multi-model responses
    """

    def __init__(self):
        """Initialize multi-provider normalizer."""
        self.normalizer = ContextNormalizer()
        self.provider_configs = {}

    def register_provider(
        self,
        name: str,
        config: Optional[Dict] = None
    ):
        """
        Register a provider with specific configuration.

        Args:
            name: Provider name
            config: Provider-specific settings
        """
        self.provider_configs[name] = config or {}

    def normalize_input_for_provider(
        self,
        prompt: str,
        provider: str
    ) -> Dict:
        """
        Normalize input prompt for specific provider.

        Args:
            prompt: Raw prompt
            provider: Target provider name

        Returns:
            Normalized prompt with provider-specific adjustments
        """
        base_normalized = self.normalizer.normalize_input(prompt)

        # Apply provider-specific adjustments
        config = self.provider_configs.get(provider, {})

        if provider == 'ollama':
            # Ollama handles longer contexts well
            base_normalized['provider'] = 'ollama'
            base_normalized['optimized_for'] = 'ollama'

        elif provider == 'openai':
            # OpenAI prefers more structured prompts
            base_normalized['provider'] = 'openai'
            base_normalized['optimized_for'] = 'openai'

        elif provider == 'anthropic':
            # Anthropic Claude prefers natural language
            base_normalized['provider'] = 'anthropic'
            base_normalized['optimized_for'] = 'anthropic'

        return base_normalized

    def normalize_outputs_from_providers(
        self,
        responses: List[Dict]
    ) -> List[Dict]:
        """
        Normalize outputs from multiple providers.

        Args:
            responses: List of dicts with 'provider', 'model', 'response' keys

        Returns:
            List of normalized responses
        """
        normalized_responses = []

        for resp_info in responses:
            provider = resp_info.get('provider')
            model = resp_info.get('model')
            response = resp_info.get('response', '')

            normalized = self.normalizer.normalize_output(
                response,
                provider=provider,
                model=model
            )

            normalized_responses.append(normalized)

        return normalized_responses
