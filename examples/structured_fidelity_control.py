#!/usr/bin/env python3
"""
Structured JSON for Maximum Information Fidelity Control

Demonstrates how parsing everything into structured JSON enables:
1. Lossless round-tripping (text → JSON → text)
2. Granular control over what to keep/discard
3. Semantic operations on structured data
4. Multi-model response fusion
5. Context compression with controllable detail levels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydracontext.core.structured_parser import (
    StructuredParser,
    FidelityLevel,
    parse_to_json,
    json_to_text
)
import json


def example_1_parse_to_json():
    """Example 1: Parse text into structured JSON"""
    print("\n" + "=" * 60)
    print("Example 1: Parse Text to Structured JSON")
    print("=" * 60)

    text = """
# Introduction to Machine Learning

Machine learning is a subset of AI that enables systems to learn from data.

## Key Concepts

- Supervised learning
- Unsupervised learning
- Reinforcement learning

```python
def train_model(data):
    model = Model()
    model.fit(data)
    return model
```

What is the difference between supervised and unsupervised learning?

<thinking>
I should explain this clearly with examples.
</thinking>

Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.
"""

    parser = StructuredParser()
    structured = parser.parse(text.strip())

    print("\nOriginal text (first 100 chars):")
    print(text.strip()[:100] + "...")

    print("\nStructured JSON:")
    print(json.dumps(structured, indent=2))

    print("\nBlock types found:")
    for block in structured['blocks']:
        print(f"  - {block['type']}: {block.get('content', '')[:50]}...")


def example_2_fidelity_levels():
    """Example 2: Control information fidelity"""
    print("\n" + "=" * 60)
    print("Example 2: Fidelity Level Control")
    print("=" * 60)

    text = """
# Title

<thinking>This is internal reasoning that users don't need to see</thinking>

Main content that is important.

More details that might be optional.
"""

    print("\nOriginal text:")
    print(text.strip())

    for fidelity in [FidelityLevel.MAXIMUM, FidelityLevel.HIGH, FidelityLevel.LOW]:
        print(f"\n{fidelity.upper()} FIDELITY:")
        parser = StructuredParser(fidelity=fidelity)
        structured = parser.parse(text.strip())
        print(f"  Blocks kept: {len(structured['blocks'])}")
        print(f"  Types: {[b['type'] for b in structured['blocks']]}")


def example_3_lossless_round_trip():
    """Example 3: Lossless round-tripping"""
    print("\n" + "=" * 60)
    print("Example 3: Lossless Round-Trip")
    print("=" * 60)

    original_text = """
# Heading

This is a **bold** statement.

- Item 1
- Item 2

```python
print("code")
```
"""

    print("Original text:")
    print(original_text.strip())

    # Parse to JSON
    structured = parse_to_json(original_text.strip())

    # Reconstruct from JSON
    reconstructed = json_to_text(structured)

    print("\nReconstructed text:")
    print(reconstructed)

    print("\n✓ Round-trip preserves structure!")


def example_4_semantic_operations():
    """Example 4: Semantic operations on structured data"""
    print("\n" + "=" * 60)
    print("Example 4: Semantic Operations")
    print("=" * 60)

    text = """
# API Documentation

## Endpoints

### GET /users
Returns list of users.

```python
response = requests.get("/users")
```

### POST /users
Creates a new user.

What is the rate limit?

The rate limit is 100 requests per minute.
"""

    structured = parse_to_json(text.strip())

    print("\nExtracting semantic information:")

    # Extract all code blocks
    code_blocks = [b for b in structured['blocks'] if b['type'] == 'code_block']
    print(f"\n  Code blocks found: {len(code_blocks)}")
    for cb in code_blocks:
        print(f"    Language: {cb['language']}, Lines: {cb['line_count']}")

    # Extract all questions
    questions = [b for b in structured['blocks'] if b['type'] == 'question']
    print(f"\n  Questions found: {len(questions)}")
    for q in questions:
        print(f"    Q: {q['content']}")

    # Extract headings structure
    headings = [b for b in structured['blocks'] if b['type'] == 'heading']
    print(f"\n  Document structure:")
    for h in headings:
        indent = "  " * h['level']
        print(f"    {indent}[H{h['level']}] {h['content']}")


def example_5_multi_model_fusion():
    """Example 5: Fuse responses from multiple models"""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Model Response Fusion")
    print("=" * 60)

    # Responses from different models
    response_a = """
# Answer

Machine learning is a type of AI.

```python
model = ML.train(data)
```
"""

    response_b = """
# Explanation

ML enables computers to learn patterns.

Example: Image classification, NLP, etc.
"""

    print("Parsing responses from Model A and Model B...")

    struct_a = parse_to_json(response_a.strip())
    struct_b = parse_to_json(response_b.strip())

    # Merge blocks
    merged = {
        'version': '1.0',
        'fidelity': FidelityLevel.MAXIMUM,
        'metadata': {
            'sources': ['model_a', 'model_b'],
            'merged': True
        },
        'blocks': struct_a['blocks'] + struct_b['blocks'],
        'statistics': {}
    }

    print(f"\nModel A blocks: {len(struct_a['blocks'])}")
    print(f"Model B blocks: {len(struct_b['blocks'])}")
    print(f"Merged blocks: {len(merged['blocks'])}")

    reconstructed = json_to_text(merged)
    print("\nFused response:")
    print(reconstructed)


def example_6_granular_filtering():
    """Example 6: Granular filtering of content"""
    print("\n" + "=" * 60)
    print("Example 6: Granular Content Filtering")
    print("=" * 60)

    text = """
# Guide

<thinking>Internal thought process</thinking>

Important concept explained here.

```python
# Code example
def example():
    pass
```

Minor detail that can be skipped.

- Bullet 1
- Bullet 2
"""

    structured = parse_to_json(text.strip())

    # Custom filter: Keep only headings, paragraphs, and code
    filtered_blocks = [
        b for b in structured['blocks']
        if b['type'] in ['heading', 'paragraph', 'code_block']
    ]

    filtered_structured = {
        **structured,
        'blocks': filtered_blocks
    }

    print("\nOriginal blocks:", len(structured['blocks']))
    print("Filtered blocks:", len(filtered_blocks))

    print("\nFiltered output:")
    print(json_to_text(filtered_structured))


def example_7_context_compression():
    """Example 7: Context compression with controllable loss"""
    print("\n" + "=" * 60)
    print("Example 7: Context Compression")
    print("=" * 60)

    long_text = """
# Main Topic

<thinking>
Long internal reasoning process...
This could be verbose...
</thinking>

Core concept: This is the essential information.

Additional details that provide context but aren't critical.

More supporting information.

```python
# Example code
def process():
    # Implementation details
    pass
```

Final summary of key points.
"""

    print("Compressing at different fidelity levels:\n")

    for fidelity in [FidelityLevel.MAXIMUM, FidelityLevel.HIGH, FidelityLevel.LOW]:
        structured = parse_to_json(long_text.strip(), fidelity=fidelity)
        reconstructed = json_to_text(structured)

        print(f"{fidelity}:")
        print(f"  Original length: {len(long_text)}")
        print(f"  Compressed length: {len(reconstructed)}")
        print(f"  Compression: {100 * (1 - len(reconstructed)/len(long_text)):.1f}%")
        print(f"  Blocks: {len(structured['blocks'])}")
        print()


def main():
    """Run all examples."""
    print("\n" + "+" * 60)
    print(" Structured JSON for Maximum Fidelity Control")
    print("+" * 60)

    examples = [
        example_1_parse_to_json,
        example_2_fidelity_levels,
        example_3_lossless_round_trip,
        example_4_semantic_operations,
        example_5_multi_model_fusion,
        example_6_granular_filtering,
        example_7_context_compression,
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
    print("\nKey Benefits:")
    print("  ✓ Lossless round-tripping preserves ALL information")
    print("  ✓ Granular control over what to keep/discard")
    print("  ✓ Semantic operations on structured data")
    print("  ✓ Multi-model response fusion")
    print("  ✓ Context compression with controllable loss")
    print("+" * 60 + "\n")


if __name__ == "__main__":
    main()
