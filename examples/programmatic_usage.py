#!/usr/bin/env python3
"""
Examples of using HydraContext programmatically in your application.

Run this file to see various usage patterns:
    python examples/programmatic_usage.py
"""

# Add parent directory to path for local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydracontext.core import (
    HydraContext,
    normalize,
    classify,
    chunk,
    quick_process,
    load_and_process
)


def example_1_basic_usage():
    """Example 1: Basic prompt processing."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Prompt Processing")
    print("=" * 60)

    # Initialize HydraContext
    hydra = HydraContext()

    # Process a single prompt
    prompt = "Explain how transformer architecture works in modern LLMs."
    result = hydra.process(prompt)

    print(f"\nInput: {prompt}")
    print(f"\nProcessed into {len(result)} segment(s)")
    print(f"Type: {result[0]['type']}")
    print(f"Token estimate: {result[0]['token_estimate']}")
    print(f"Hash: {result[0]['hash'][:16]}...")


def example_2_batch_processing():
    """Example 2: Batch processing with deduplication."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing with Deduplication")
    print("=" * 60)

    hydra = HydraContext()

    prompts = [
        "What is machine learning?",
        "Explain deep learning",
        "What is machine learning?",  # Duplicate
        "How do neural networks work?",
    ]

    print(f"\nProcessing {len(prompts)} prompts...")
    results = hydra.process_batch(prompts)

    print(f"\nResults:")
    for i, segments in enumerate(results):
        duplicate = segments[0].get('duplicate', False)
        status = "DUPLICATE" if duplicate else "UNIQUE"
        print(f"  Prompt {i+1}: {status} - {segments[0]['type']}")

    stats = hydra.stats()
    print(f"\nStatistics:")
    print(f"  Total processed: {stats['processed_count']}")
    print(f"  Unique prompts: {stats['unique_hashes']}")


def example_3_code_classification():
    """Example 3: Automatic code detection and classification."""
    print("\n" + "=" * 60)
    print("Example 3: Code Classification")
    print("=" * 60)

    code_prompt = """
Review this Python code:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Suggest improvements.
"""

    hydra = HydraContext()
    result = hydra.process(code_prompt)

    print(f"Type detected: {result[0]['type']}")
    if 'code_blocks' in result[0]:
        print(f"Code blocks found: {len(result[0]['code_blocks'])}")
        for block in result[0]['code_blocks']:
            print(f"  Language: {block['language']}")


def example_4_convenience_functions():
    """Example 4: Using convenience functions."""
    print("\n" + "=" * 60)
    print("Example 4: Convenience Functions")
    print("=" * 60)

    # Normalize messy prompt
    messy = "   What    is   AI?   \n\n\n\n   Explain it.   "
    clean = normalize(messy)
    print(f"\nNormalized:")
    print(f"  Before: {repr(messy[:30])}...")
    print(f"  After:  {repr(clean)}")

    # Classify prompts
    prompts = [
        "Explain how LLMs work",
        "```python\ndef foo(): pass```",
        "User: Hello\nAssistant: Hi!",
    ]

    print(f"\nClassification:")
    for p in prompts:
        prompt_type = classify(p)
        print(f"  {p[:30]:30s} -> {prompt_type}")

    # Chunk long text
    long_text = "Lorem ipsum " * 500  # ~6000 chars
    chunks = chunk(long_text, max_chars=1024)
    print(f"\nChunking:")
    print(f"  Original length: {len(long_text)} chars")
    print(f"  Split into: {len(chunks)} chunks")
    print(f"  Chunk sizes: {[len(c) for c in chunks]}")


def example_5_quick_process():
    """Example 5: Quick one-off processing."""
    print("\n" + "=" * 60)
    print("Example 5: Quick Process")
    print("=" * 60)

    prompt = "Explain the difference between supervised and unsupervised learning."

    result = quick_process(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"\nQuick Summary:")
    print(f"  Type: {result['type']}")
    print(f"  Segments: {result['segments']}")
    print(f"  Total length: {result['total_length']} chars")
    print(f"  Token estimate: {result['token_estimate']}")


def example_6_integration_pattern():
    """Example 6: Integration into an LLM application."""
    print("\n" + "=" * 60)
    print("Example 6: Integration Pattern for LLM Application")
    print("=" * 60)

    # Simulating an LLM application
    class MyLLMApp:
        def __init__(self):
            self.context = HydraContext(max_chars=4096)
            self.conversation_history = []

        def process_user_input(self, user_prompt: str) -> dict:
            """Process and prepare user input for LLM."""
            # Normalize and process the prompt
            segments = self.context.process(
                user_prompt,
                prompt_id=f"user_{len(self.conversation_history)}"
            )

            # Extract the main content
            processed = segments[0]

            # Check if it's a duplicate (user repeated themselves)
            if processed.get('duplicate', False):
                return {
                    'status': 'duplicate',
                    'message': 'This seems similar to a previous message.'
                }

            # Store in history
            self.conversation_history.append({
                'user_input': user_prompt,
                'processed': processed,
                'type': processed['type']
            })

            return {
                'status': 'success',
                'type': processed['type'],
                'token_estimate': processed['token_estimate'],
                'ready_for_llm': True
            }

        def get_stats(self):
            """Get processing statistics."""
            return self.context.stats()

    # Use the app
    app = MyLLMApp()

    # Process some user inputs
    inputs = [
        "What is the capital of France?",
        "Explain quantum computing",
        "What is the capital of France?",  # Duplicate
    ]

    print("\nSimulated LLM Application:")
    for i, user_input in enumerate(inputs, 1):
        result = app.process_user_input(user_input)
        print(f"\n  Input {i}: {user_input[:40]}")
        print(f"    Status: {result['status']}")
        if result['status'] == 'success':
            print(f"    Type: {result['type']}")
            print(f"    Tokens: {result['token_estimate']}")

    print(f"\n  App Stats: {app.get_stats()}")


def example_7_load_from_file():
    """Example 7: Load and process from files."""
    print("\n" + "=" * 60)
    print("Example 7: Load from Files")
    print("=" * 60)

    # Load from text file
    text_file = Path(__file__).parent / "simple_prompt.txt"
    if text_file.exists():
        segments = load_and_process(text_file)
        print(f"\nLoaded from {text_file.name}:")
        print(f"  Segments: {len(segments)}")
        print(f"  Type: {segments[0]['type']}")

    # Load from JSON
    json_file = Path(__file__).parent / "prompts.json"
    if json_file.exists():
        segments = load_and_process(json_file)
        print(f"\nLoaded from {json_file.name}:")
        print(f"  Total segments: {len(segments)}")
        types = set(s['type'] for s in segments)
        print(f"  Types found: {types}")


def example_8_stateful_caching():
    """Example 8: Stateful caching across sessions."""
    print("\n" + "=" * 60)
    print("Example 8: Stateful Caching")
    print("=" * 60)

    hydra = HydraContext()

    # First batch
    print("\nProcessing first batch...")
    batch1 = ["What is AI?", "Explain ML", "What is DL?"]
    hydra.process_batch(batch1)
    print(f"  Stats: {hydra.stats()}")

    # Second batch with some duplicates
    print("\nProcessing second batch (with duplicates)...")
    batch2 = ["What is AI?", "New question", "Explain ML"]
    hydra.process_batch(batch2)
    print(f"  Stats: {hydra.stats()}")

    # Reset and start fresh
    print("\nResetting cache...")
    hydra.reset()
    print(f"  Stats after reset: {hydra.stats()}")


def main():
    """Run all examples."""
    print("\n" + "+" * 60)
    print(" HydraContext - Programmatic Usage Examples")
    print("+" * 60)

    examples = [
        example_1_basic_usage,
        example_2_batch_processing,
        example_3_code_classification,
        example_4_convenience_functions,
        example_5_quick_process,
        example_6_integration_pattern,
        example_7_load_from_file,
        example_8_stateful_caching,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")

    print("\n" + "+" * 60)
    print(" Examples Complete!")
    print("+" * 60 + "\n")


if __name__ == "__main__":
    main()
