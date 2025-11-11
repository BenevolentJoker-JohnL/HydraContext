#!/usr/bin/env python3
"""
Integration test to verify HydraContext works in applications.
Run this to verify everything is working correctly.
"""

from hydracontext.api import HydraContextAPI
from hydracontext import (
    ContextSegmenter,
    ContentClassifier,
    ContentDeduplicator
)


def test_high_level_api():
    """Test the high-level API - recommended for most applications."""
    print("\n" + "=" * 70)
    print("TEST 1: High-Level API (HydraContextAPI)")
    print("=" * 70)

    # Initialize
    hydra = HydraContextAPI(
        max_chars=2048,
        overlap=200,
        auto_deduplicate=True
    )

    # Test 1: Single prompt processing
    prompt = "Explain how transformer architecture works in modern LLMs."
    result = hydra.process(prompt)

    print(f"\n✓ Processed single prompt")
    print(f"  Type: {result[0]['type']}")
    print(f"  Token estimate: {result[0]['token_estimate']}")
    print(f"  Content length: {result[0]['length']} chars")

    # Test 2: Batch processing with deduplication
    prompts = [
        "What is machine learning?",
        "Explain deep learning",
        "What is machine learning?",  # Duplicate
        "How do neural networks work?",
    ]

    results = hydra.process_batch(prompts)
    print(f"\n✓ Processed batch of {len(prompts)} prompts")

    duplicates_found = sum(1 for segments in results
                          if segments[0].get('duplicate', False))
    print(f"  Duplicates detected: {duplicates_found}")

    # Test 3: Statistics
    stats = hydra.stats()
    print(f"\n✓ Statistics:")
    print(f"  Total processed: {stats['processed_count']}")
    print(f"  Unique items: {stats['unique_hashes']}")

    return True


def test_low_level_components():
    """Test individual components - for advanced use cases."""
    print("\n" + "=" * 70)
    print("TEST 2: Low-Level Components")
    print("=" * 70)

    # Initialize components
    segmenter = ContextSegmenter()
    classifier = ContentClassifier()
    deduplicator = ContentDeduplicator()

    # Test text with mixed content
    text = """
    Python is a versatile programming language.
    Here's a simple example:

    ```python
    def greet(name):
        return f"Hello, {name}!"
    ```

    This function demonstrates basic string formatting.
    """

    # Segment the text
    segments = segmenter.segment_text(text, granularity='sentence')
    print(f"\n✓ Segmented text into {len(segments)} segments")

    # Process each segment
    unique_count = 0
    for i, segment in enumerate(segments, 1):
        # Classify content
        classification = classifier.classify(segment.text)

        # Check for duplicates
        is_dup = deduplicator.is_duplicate(segment.text, record=True)

        if not is_dup:
            unique_count += 1
            print(f"\n  Segment {i}:")
            print(f"    Type: {classification.content_type.value}")
            print(f"    Confidence: {classification.confidence:.1%}")
            print(f"    Length: {len(segment.text)} chars")

    print(f"\n✓ Unique segments: {unique_count}/{len(segments)}")

    # Get deduplication stats
    stats = deduplicator.get_statistics()
    print(f"\n✓ Deduplication stats:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Unique: {stats['unique_content']}")
    print(f"  Duplicates: {stats['duplicates_found']}")

    return True


def test_code_detection():
    """Test code vs prose detection."""
    print("\n" + "=" * 70)
    print("TEST 3: Code vs Prose Detection")
    print("=" * 70)

    classifier = ContentClassifier()

    test_cases = [
        ("This is a plain English sentence.", "prose"),
        ("def foo():\n    return 42", "code"),
        ('{"key": "value"}', "structured_data"),
        ("User: Hello\nAssistant: Hi!", "prose"),
    ]

    correct = 0
    for text, expected_type in test_cases:
        result = classifier.classify(text)
        actual_type = result.content_type.value
        match = "✓" if expected_type in actual_type or actual_type in expected_type else "✗"

        print(f"\n  {match} Text: {text[:40]}")
        print(f"     Detected: {actual_type} (confidence: {result.confidence:.1%})")

        if expected_type in actual_type or actual_type in expected_type:
            correct += 1

    print(f"\n✓ Accuracy: {correct}/{len(test_cases)} correct")
    return correct >= len(test_cases) - 1  # Allow 1 mistake


def test_application_integration():
    """Simulate real application usage."""
    print("\n" + "=" * 70)
    print("TEST 4: Application Integration Pattern")
    print("=" * 70)

    class SimpleLLMApp:
        """Example LLM application using HydraContext."""

        def __init__(self):
            self.hydra = HydraContextAPI(max_chars=4096)
            self.message_history = []

        def process_message(self, user_input: str) -> dict:
            """Process user message before sending to LLM."""
            segments = self.hydra.process(user_input)
            processed = segments[0]

            # Check if duplicate (user repeating themselves)
            if processed.get('duplicate', False):
                return {
                    'status': 'duplicate',
                    'message': 'Similar to previous input'
                }

            # Store and prepare for LLM
            self.message_history.append({
                'input': user_input,
                'type': processed['type'],
                'tokens': processed['token_estimate']
            })

            return {
                'status': 'ready',
                'type': processed['type'],
                'tokens': processed['token_estimate']
            }

        def get_stats(self):
            return self.hydra.stats()

    # Use the app
    app = SimpleLLMApp()

    messages = [
        "What is the capital of France?",
        "Explain quantum computing briefly",
        "What is the capital of France?",  # Duplicate
    ]

    for i, msg in enumerate(messages, 1):
        result = app.process_message(msg)
        status = "✓" if result['status'] == 'ready' else "⚠"
        print(f"\n  {status} Message {i}: {msg[:40]}")
        print(f"     Status: {result['status']}")
        if result['status'] == 'ready':
            print(f"     Type: {result['type']}")
            print(f"     Tokens: {result['tokens']}")

    stats = app.get_stats()
    print(f"\n✓ App stats: {stats['processed_count']} processed, "
          f"{stats['unique_hashes']} unique")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "█" * 70)
    print("  HydraContext - Integration Tests")
    print("█" * 70)

    tests = [
        ("High-Level API", test_high_level_api),
        ("Low-Level Components", test_low_level_components),
        ("Code Detection", test_code_detection),
        ("Application Integration", test_application_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n⚠ {name} had issues")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} failed: {e}")

    # Summary
    print("\n" + "█" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("█" * 70 + "\n")

    if failed == 0:
        print("✓ All tests passed! HydraContext is working correctly.\n")
        return 0
    else:
        print(f"⚠ {failed} test(s) had issues. Check output above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
