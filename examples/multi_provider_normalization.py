#!/usr/bin/env python3
"""
Multi-Provider Response Normalization Example

Demonstrates how HydraContext normalizes outputs from different LLM providers
into a unified JSON format.

This enables:
- Consistent output format across OpenAI, Anthropic, Ollama, etc.
- Easy provider switching
- Multi-model response aggregation
- Bidirectional normalization (input + output)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydracontext.core.response_processor import ResponseNormalizer
from hydracontext.core.bidirectional import ContextNormalizer
import json


def example_1_ollama_response():
    """Example 1: Normalize Ollama response"""
    print("\n" + "=" * 60)
    print("Example 1: Ollama Response Normalization")
    print("=" * 60)

    # Simulated Ollama API response
    ollama_response = {
        "model": "llama2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "done": True,
        "prompt_eval_count": 26,
        "eval_count": 42,
        "total_duration": 4883583458,
    }

    normalizer = ResponseNormalizer()
    normalized = normalizer.normalize_json_response(ollama_response, provider='ollama')

    print("\nOriginal Ollama format (partial):")
    print(f"  model: {ollama_response['model']}")
    print(f"  response: {ollama_response['response'][:50]}...")

    print("\nNormalized to unified format:")
    print(f"  provider: {normalized['provider']}")
    print(f"  model: {normalized['model']}")
    print(f"  content: {normalized['content'][:50]}...")
    print(f"  usage.prompt_tokens: {normalized['usage']['prompt_tokens']}")
    print(f"  usage.completion_tokens: {normalized['usage']['completion_tokens']}")


def example_2_openai_response():
    """Example 2: Normalize OpenAI response"""
    print("\n" + "=" * 60)
    print("Example 2: OpenAI Response Normalization")
    print("=" * 60)

    # Simulated OpenAI API response
    openai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 25,
            "total_tokens": 34
        }
    }

    normalizer = ResponseNormalizer()
    normalized = normalizer.normalize_json_response(openai_response, provider='openai')

    print("\nOriginal OpenAI format (partial):")
    print(f"  model: {openai_response['model']}")
    print(f"  choices[0].message.content: {openai_response['choices'][0]['message']['content'][:50]}...")

    print("\nNormalized to unified format:")
    print(f"  provider: {normalized['provider']}")
    print(f"  model: {normalized['model']}")
    print(f"  content: {normalized['content'][:50]}...")
    print(f"  usage.prompt_tokens: {normalized['usage']['prompt_tokens']}")
    print(f"  usage.completion_tokens: {normalized['usage']['completion_tokens']}")


def example_3_anthropic_response():
    """Example 3: Normalize Anthropic response"""
    print("\n" + "=" * 60)
    print("Example 3: Anthropic Response Normalization")
    print("=" * 60)

    # Simulated Anthropic API response
    anthropic_response = {
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience."
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 25
        }
    }

    normalizer = ResponseNormalizer()
    normalized = normalizer.normalize_json_response(anthropic_response, provider='anthropic')

    print("\nOriginal Anthropic format (partial):")
    print(f"  model: {anthropic_response['model']}")
    print(f"  content[0].text: {anthropic_response['content'][0]['text'][:50]}...")

    print("\nNormalized to unified format:")
    print(f"  provider: {normalized['provider']}")
    print(f"  model: {normalized['model']}")
    print(f"  content: {normalized['content'][:50]}...")
    print(f"  usage.prompt_tokens: {normalized['usage']['prompt_tokens']}")
    print(f"  usage.completion_tokens: {normalized['usage']['completion_tokens']}")


def example_4_compare_all_providers():
    """Example 4: Compare normalized responses from all providers"""
    print("\n" + "=" * 60)
    print("Example 4: Unified Format Across All Providers")
    print("=" * 60)

    responses = {
        'ollama': {
            "model": "llama2",
            "response": "AI is the simulation of human intelligence.",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 15,
        },
        'openai': {
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "AI is the simulation of human intelligence."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        },
        'anthropic': {
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "AI is the simulation of human intelligence."}],
            "usage": {"input_tokens": 10, "output_tokens": 15}
        }
    }

    normalizer = ResponseNormalizer()

    print("\nAll providers return the same unified format:\n")

    for provider, response in responses.items():
        normalized = normalizer.normalize_json_response(response, provider=provider)
        print(f"{provider.upper()}:")
        print(f"  content: {normalized['content']}")
        print(f"  usage: {normalized['usage']}")
        print(f"  model: {normalized['model']}")
        print()


def example_5_bidirectional_normalization():
    """Example 5: Full bidirectional normalization (input + output)"""
    print("\n" + "=" * 60)
    print("Example 5: Bidirectional Normalization")
    print("=" * 60)

    context_normalizer = ContextNormalizer()

    # Step 1: Normalize INPUT (prompt)
    user_prompt = "   What is   machine learning?  \n\n\n  Explain it simply.   "
    normalized_input = context_normalizer.normalize_input(user_prompt)

    print("\nINPUT NORMALIZATION:")
    print(f"Original prompt: {repr(user_prompt[:30])}...")
    print(f"Normalized prompt: {normalized_input['content']}")
    print(f"Type: {normalized_input['type']}")
    print(f"Token estimate: {normalized_input['token_estimate']}")

    # Step 2: Simulate sending to LLM and getting response
    print("\n[Sending to LLM...]")

    # Step 3: Normalize OUTPUT (response from Ollama)
    ollama_response = {
        "model": "llama2",
        "response": "<thinking>Let me explain this clearly</thinking>\n\nMachine learning enables computers to learn from data.",
        "done": True,
        "prompt_eval_count": 15,
        "eval_count": 20,
    }

    normalized_output = context_normalizer.normalize_ollama_output(ollama_response)

    print("\nOUTPUT NORMALIZATION:")
    print(f"Original response: {ollama_response['response'][:60]}...")
    print(f"Normalized (thinking removed): {normalized_output['content']}")
    print(f"Usage: {normalized_output['usage']}")


def example_6_provider_agnostic_workflow():
    """Example 6: Provider-agnostic application code"""
    print("\n" + "=" * 60)
    print("Example 6: Provider-Agnostic LLM Application")
    print("=" * 60)

    def process_llm_response(raw_response, provider):
        """
        Generic function that works with ANY provider.

        This is the power of HydraContext - your application code
        doesn't need to know about provider-specific formats.
        """
        normalizer = ResponseNormalizer()
        normalized = normalizer.normalize_json_response(raw_response, provider=provider)

        return {
            'text': normalized['content'],
            'tokens_used': normalized['usage']['total_tokens'],
            'model': normalized['model']
        }

    # Works with Ollama
    ollama_resp = {"model": "llama2", "response": "Hello!", "done": True, "eval_count": 5}
    result1 = process_llm_response(ollama_resp, 'ollama')
    print(f"\nOllama result: {result1}")

    # Works with OpenAI
    openai_resp = {
        "model": "gpt-4",
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"total_tokens": 10}
    }
    result2 = process_llm_response(openai_resp, 'openai')
    print(f"OpenAI result: {result2}")

    # Same function, different providers - no code changes needed!
    print("\n✓ Same application code works with all providers")


def example_7_auto_detection():
    """Example 7: Automatic provider detection"""
    print("\n" + "=" * 60)
    print("Example 7: Automatic Provider Detection")
    print("=" * 60)

    normalizer = ResponseNormalizer()

    responses = [
        {"model": "llama2", "response": "Text", "done": True},  # Ollama
        {"choices": [{"message": {"content": "Text"}}]},  # OpenAI
        {"content": [{"type": "text", "text": "Text"}]},  # Anthropic
    ]

    print("\nAuto-detecting providers from response format:\n")

    for resp in responses:
        # No provider specified - auto-detect!
        normalized = normalizer.normalize_json_response(resp)
        print(f"Detected: {normalized['provider']}")


def main():
    """Run all examples."""
    print("\n" + "+" * 60)
    print(" Multi-Provider Response Normalization Examples")
    print("+" * 60)

    examples = [
        example_1_ollama_response,
        example_2_openai_response,
        example_3_anthropic_response,
        example_4_compare_all_providers,
        example_5_bidirectional_normalization,
        example_6_provider_agnostic_workflow,
        example_7_auto_detection,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "+" * 60)
    print(" Examples Complete!")
    print("+" * 60)
    print("\nKey Takeaway:")
    print("  HydraContext normalizes ALL provider responses to a unified")
    print("  JSON format, making your code provider-agnostic!")
    print("+" * 60 + "\n")


if __name__ == "__main__":
    main()
