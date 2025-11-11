# HydraContext Quick Reference Guide

## Installation & Basic Setup

```python
from hydracontext.core.bidirectional import ContextNormalizer
from hydracontext.core.response_processor import ResponseNormalizer
from hydracontext.core.prompt_processor import PromptProcessor
from hydracontext.core.structured_parser import StructuredParser, FidelityLevel
from hydracontext.core.segmenter import ContextSegmenter
from hydracontext.core.classifier import ContentClassifier
from hydracontext.core.deduplicator import ContentDeduplicator
```

---

## Core API Reference

### 1. Bidirectional Normalization (Recommended Entry Point)

```python
from hydracontext.core.bidirectional import ContextNormalizer

normalizer = ContextNormalizer(
    max_chars=2048,           # Max prompt segment size
    overlap=200,              # Overlap between segments
    remove_thinking=True,     # Remove thinking blocks
    normalize_code_blocks=True
)

# Normalize input (prompt)
prompt_data = normalizer.normalize_input("Your prompt here")
print(prompt_data['content'])          # Clean prompt
print(prompt_data['type'])             # detection result
print(prompt_data['token_estimate'])   # Estimated tokens

# Normalize output (response)
response_data = normalizer.normalize_output(
    response_text,
    provider='ollama',
    model='llama2'
)
print(response_data['normalized_content'])
print(response_data['reduction_ratio'])  # Artifact removal %
print(response_data['metadata'])         # Structure info
```

---

### 2. Prompt Processing

```python
from hydracontext.core.prompt_processor import (
    PromptProcessor,
    normalize_prompt,
    detect_prompt_type,
    split_prompt,
    deduplicate_prompts
)

# High-level API
processor = PromptProcessor(max_chars=2048, overlap=200)
segments = processor.process("Your prompt", prompt_id="custom_id")
batch_results = processor.process_batch([prompt1, prompt2, ...])
stats = processor.get_statistics()

# Low-level functions
normalized = normalize_prompt(raw_prompt)
prompt_type = detect_prompt_type(text)  # Returns: code/conversation/example/instruction/system
segments = split_prompt(text, max_chars=2048, overlap=200)
marked_segments, hashes = deduplicate_prompts(segments)
```

**Segment Output Format**:
```python
{
    "id": "prompt_0_seg_0",
    "prompt_id": "prompt_0",
    "segment_index": 0,
    "type": "instruction",
    "normalized": True,
    "content": "...",
    "length": 250,
    "hash": "abc123...",
    "duplicate": False,
    "token_estimate": 62,
    "timestamp": "2024-01-01T12:00:00.000000",
    "code_blocks": [{"language": "python", "content": "..."}]  # if present
}
```

---

### 3. Response Processing

```python
from hydracontext.core.response_processor import (
    ResponseNormalizer,
    StreamingNormalizer,
    ResponseComparator,
    OllamaResponseHandler
)

# Basic normalization
normalizer = ResponseNormalizer(
    remove_thinking=True,
    remove_system=True,
    normalize_code_blocks=True,
    strip_artifacts=True
)

# Plain text response
clean = normalizer.normalize(raw_response, provider='ollama', model='llama2')

# Provider API response (JSON)
openai_resp = {"choices": [{"message": {"content": "..."}}], "usage": {...}}
normalized = normalizer.normalize_json_response(openai_resp, provider='openai')

# Streaming responses
streaming_normalizer = StreamingNormalizer(normalizer)
for chunk in stream:
    normalized_chunk = streaming_normalizer.process_chunk(chunk)
    if normalized_chunk:
        print(normalized_chunk)
final = streaming_normalizer.flush()

# Compare multiple responses
comparator = ResponseComparator()
comparison = comparator.compare(
    [response1, response2, response3],
    similarity_threshold=0.8
)

# Ollama-specific handling
ollama_handler = OllamaResponseHandler()
ollama_normalized = ollama_handler.normalize_ollama_response(ollama_dict)
for normalized_chunk in ollama_handler.process_streaming_response(stream):
    print(normalized_chunk)
```

**Normalized Response Format**:
```python
{
    "normalized_content": "...",
    "original_content": "...",
    "provider": "ollama",
    "model": "llama2",
    "original_length": 1500,
    "normalized_length": 1450,
    "reduction_ratio": 0.033,
    "token_estimate": 362,
    "metadata": {
        "has_code": True,
        "code_blocks": 2,
        "has_lists": True,
        "has_headings": False,
        "line_count": 45,
        "paragraph_count": 8
    },
    "hash": "xyz789...",
    "timestamp": "2024-01-01T12:00:00"
}
```

---

### 4. Provider Parsers (Auto-Detection)

```python
from hydracontext.core.provider_parsers import UnifiedResponseParser

parser = UnifiedResponseParser()

# Explicit provider
parsed = parser.parse(raw_response, provider='openai')

# Auto-detect provider
parsed = parser.parse(raw_response)  # Detects OpenAI/Anthropic/Ollama/Generic

# Standard output format
print(parsed['content'])           # Extracted text
print(parsed['provider'])          # Provider name
print(parsed['model'])             # Model name
print(parsed['usage'])             # Token counts (normalized)
print(parsed['finish_reason'])     # stop/length/error/etc
print(parsed['metadata'])          # Provider-specific data
```

---

### 5. Structured Parsing

```python
from hydracontext.core.structured_parser import (
    StructuredParser,
    FidelityLevel,
    parse_to_json,
    json_to_text
)

# Create parser with fidelity level
parser = StructuredParser(fidelity=FidelityLevel.MAXIMUM)
# Options: MAXIMUM, HIGH, MEDIUM, LOW, MINIMAL

# Parse text to structured JSON
structured = parser.parse(text, metadata={'source': 'user'})

# Examine blocks
for block in structured['blocks']:
    print(f"Type: {block['type']}")
    print(f"Content: {block.get('content', '')}")
    if 'inline_formatting' in block:
        print(f"Formatting: {block['inline_formatting']}")

# Reconstruct text
reconstructed = parser.reconstruct(structured)

# Convenience functions
structured = parse_to_json(text, fidelity=FidelityLevel.HIGH)
text = json_to_text(structured)

# Statistics
stats = structured['statistics']
print(f"Total blocks: {stats['total_blocks']}")
print(f"Has code: {stats['has_code']}")
print(f"Has lists: {stats['has_lists']}")
```

**Block Types**:
- `heading` (with level: 1-6)
- `paragraph`
- `code_block` (with language)
- `list` (unordered or ordered)
- `question`
- `thinking`

---

### 6. Text Segmentation

```python
from hydracontext.core.segmenter import ContextSegmenter, SegmentType

segmenter = ContextSegmenter(
    min_sentence_length=3,
    preserve_code=True
)

# Segment by sentences
sentences = segmenter.segment_sentences(text)
for seg in sentences:
    print(f"{seg.type}: {seg.text}")
    print(f"Position: {seg.start_pos}-{seg.end_pos}")

# Segment by paragraphs
paragraphs = segmenter.segment_paragraphs(text)

# Generic segmentation
segments = segmenter.segment_text(text, granularity='sentence')
# or
segments = segmenter.segment_text(text, granularity='paragraph')

# Segment attributes
seg.text          # Content
seg.type          # SENTENCE, PARAGRAPH, CODE_BLOCK, LIST_ITEM, HEADING
seg.start_pos     # Position in original
seg.end_pos       # End position
seg.metadata      # Additional info
```

---

### 7. Content Classification

```python
from hydracontext.core.classifier import ContentClassifier, ContentType

classifier = ContentClassifier(threshold=0.6)

# Classify text
result = classifier.classify(text)

# Access results
print(f"Type: {result.content_type}")           # CODE, PROSE, STRUCTURED_DATA, MIXED, UNKNOWN
print(f"Confidence: {result.confidence}")       # 0.0-1.0
print(f"Indicators: {result.indicators}")       # Individual feature scores
print(f"Metadata: {result.metadata}")           # Additional info

# Available indicators
indicators = {
    'code_syntax': 0.8,        # Bracket/operator density
    'code_keywords': 0.75,     # Programming keywords
    'indentation': 0.9,        # Consistent indentation
    'prose_patterns': 0.2,     # Natural language patterns
    'structured_data': 0.0,    # JSON/XML/YAML
    'punctuation': 0.15,       # Sentence punctuation
    'line_length': 0.3,        # Average line length
    'whitespace': 0.7          # Whitespace density
}
```

---

### 8. Content Deduplication

```python
from hydracontext.core.deduplicator import ContentDeduplicator
from pathlib import Path

# Create deduplicator with persistent cache
dedup = ContentDeduplicator(
    algorithm='sha256',        # md5, sha256, blake2b
    normalize=True,
    cache_path=Path('cache.jsonl'),
    min_length=10
)

# Check for duplicates
is_dup = dedup.is_duplicate(text, record=True)

# Deduplicate list
unique_texts = dedup.deduplicate_list([text1, text2, text3])

# Get info about text
hash_info = dedup.get_hash_info(text)
if hash_info:
    print(f"First seen: {hash_info.first_seen}")
    print(f"Occurrences: {hash_info.occurrences}")

# Statistics
stats = dedup.get_statistics()
print(stats)
# {
#     'total_processed': 1000,
#     'unique_content': 850,
#     'duplicates_found': 150,
#     'cache_hits': 45,
#     'unique_hashes': 850,
#     'dedup_ratio': 0.15
# }

# Persist cache
dedup.save_cache(Path('hashes.jsonl'))

# Export hashes
dedup.export_hashes(Path('hashes.csv'), format='csv')
```

---

## Common Patterns

### Pattern 1: Full Pipeline with OpenAI

```python
from hydracontext.core.bidirectional import ContextNormalizer
from openai import OpenAI

normalizer = ContextNormalizer()
client = OpenAI()

# Normalize input
prompt_data = normalizer.normalize_input("Explain quantum computing")
clean_prompt = prompt_data['content']

# Call OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": clean_prompt}]
)

# Normalize output
response_data = normalizer.normalize_output(
    response.choices[0].message.content,
    provider='openai',
    model='gpt-4'
)

print(f"Clean response: {response_data['normalized_content']}")
print(f"Reduction: {response_data['reduction_ratio']:.1%}")
```

### Pattern 2: Batch Processing with Statistics

```python
from hydracontext.core.prompt_processor import PromptProcessor
import json

processor = PromptProcessor(max_chars=2048, overlap=200)

# Process batch
prompts = [
    "First prompt...",
    "Second prompt...",
    {"id": "custom", "content": "Third prompt..."}
]

results = processor.process_batch(prompts)

# Save results
output = {
    "metadata": processor.get_statistics(),
    "results": results
}

with open('output.json', 'w') as f:
    json.dump(output, f, indent=2)
```

### Pattern 3: Multi-Model Response Comparison

```python
from hydracontext.core.response_processor import ResponseNormalizer, ResponseComparator

normalizer = ResponseNormalizer()
comparator = ResponseComparator()

# Normalize responses from different models
responses = [
    normalizer.normalize(openai_response, provider='openai', model='gpt-4'),
    normalizer.normalize(anthropic_response, provider='anthropic', model='claude-3'),
    normalizer.normalize(ollama_response, provider='ollama', model='llama2')
]

# Compare
comparison = comparator.compare(responses, similarity_threshold=0.8)

print(f"Unique responses: {comparison['unique_responses']}")
print(f"Duplicates: {comparison['duplicate_responses']}")
print(f"Average similarity: {comparison['average_similarity']:.2f}")
```

### Pattern 4: Streaming Response Processing

```python
from hydracontext.core.response_processor import OllamaResponseHandler
import requests

handler = OllamaResponseHandler()

# Stream from Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama2", "prompt": prompt, "stream": True},
    stream=True
)

for line in response.iter_lines():
    chunk = json.loads(line)
    for normalized in handler.process_streaming_response(iter([chunk])):
        print(normalized['content'], end='', flush=True)
```

### Pattern 5: Structured Analysis & Reconstruction

```python
from hydracontext.core.structured_parser import StructuredParser, FidelityLevel

parser = StructuredParser(fidelity=FidelityLevel.MAXIMUM)

# Parse response
structured = parser.parse(response_text)

# Analyze
print(f"Code blocks: {structured['statistics']['block_types'].get('code_block', 0)}")
print(f"Lists: {structured['statistics']['block_types'].get('list', 0)}")
print(f"Total length: {structured['statistics']['total_length']}")

# Extract specific content
code_blocks = [b for b in structured['blocks'] if b['type'] == 'code_block']
for block in code_blocks:
    print(f"Language: {block['language']}")
    print(f"Code: {block['content']}")

# Reconstruct at lower fidelity (removes thinking, metadata, etc)
parser.fidelity = FidelityLevel.MEDIUM
simplified = parser.reconstruct(structured)
```

---

## Configuration Reference

### PromptProcessor Options

```python
PromptProcessor(
    max_chars=2048,     # Max segment size
    overlap=200         # Overlap between segments
)
```

### ResponseNormalizer Options

```python
ResponseNormalizer(
    remove_thinking=True,           # Remove <thinking> blocks
    remove_system=True,             # Remove <system> blocks
    normalize_code_blocks=True,     # Fix code fence formatting
    strip_artifacts=True            # Remove model-specific artifacts
)
```

### ContextNormalizer Options

```python
ContextNormalizer(
    max_chars=2048,
    overlap=200,
    remove_thinking=True,
    normalize_code_blocks=True
)
```

### ContentClassifier Options

```python
ContentClassifier(
    threshold=0.6  # Confidence threshold for classification (0.0-1.0)
)
```

### ContentDeduplicator Options

```python
ContentDeduplicator(
    algorithm='sha256',           # Hash algorithm: md5, sha256, blake2b
    normalize=True,               # Normalize text before hashing
    cache_path=Path('cache.jsonl'),  # Persistent cache file
    min_length=10                 # Minimum text length to process
)
```

### ContextSegmenter Options

```python
ContextSegmenter(
    min_sentence_length=3,        # Minimum chars for valid sentence
    preserve_code=True            # Keep code blocks intact
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Provider not detected | Explicitly specify `provider='openai'` when calling normalize methods |
| Code blocks not extracted | Ensure text uses proper markdown fences: \`\`\`language code\`\`\` |
| High reduction ratio | Check `normalized_content` - artifacts like `<thinking>` tags are being removed |
| Slow deduplication | Use persistent cache with `ContentDeduplicator(cache_path=...)` |
| Accuracy issues in classification | Adjust classifier threshold or check raw `indicators` dict |
| Memory usage with large texts | Use `StreamingNormalizer` for processing chunks |

---

## Performance Tips

1. **Cache Hashes**: Use `ContentDeduplicator` with persistent cache for repeated content
2. **Streaming**: Use `StreamingNormalizer` for large or streaming responses
3. **Batch Processing**: Use `process_batch()` for multiple prompts
4. **Token Estimation**: Use built-in `count_tokens_estimate()` (4 chars = 1 token approx)
5. **Structured Parsing**: Choose appropriate `FidelityLevel` to control detail

---

## Common Questions

**Q: Which class should I use?**
A: Start with `ContextNormalizer` - it's the main entry point that combines everything.

**Q: How does it detect the provider?**
A: `UnifiedResponseParser` checks for provider-specific response structures (choices for OpenAI, content array for Anthropic, response field for Ollama).

**Q: Can I use multiple providers in the same application?**
A: Yes! Use `MultiProviderNormalizer` to register multiple providers and normalize their outputs consistently.

**Q: What's the difference between fidelity levels?**
A: MAXIMUM keeps everything, HIGH removes formatting only, MEDIUM removes auxiliary info, LOW is summary only, MINIMAL is core message.

**Q: How accurate is the content classifier?**
A: It uses 8 heuristic features with weighted scoring. Best for distinguishing code from prose. Check the `confidence` score and `indicators` for details.

**Q: Does it support streaming?**
A: Yes! Use `StreamingNormalizer` or `OllamaResponseHandler.process_streaming_response()`.

---

**Last Updated**: 2024-11-11  
**For detailed API docs**: See FEATURES_ANALYSIS.md  
**For architecture**: See ARCHITECTURE_DIAGRAM.md
