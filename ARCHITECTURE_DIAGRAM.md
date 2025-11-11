# HydraContext Architecture Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HydraContext Framework                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────┐          ┌─────────────────────┐          │
│  │  User Application   │          │  User Application   │          │
│  │  (Prompt)           │          │  (Response)         │          │
│  └──────────┬──────────┘          └────────────┬────────┘          │
│             │                                  │                    │
│             ▼                                  ▼                    │
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │  Input Normalization     │    │  Output Normalization    │      │
│  │  (PromptProcessor)       │    │  (ResponseNormalizer)    │      │
│  └──────────┬───────────────┘    └──────────┬───────────────┘      │
│             │                                │                     │
│             │ normalize_prompt()             │ normalize()         │
│             │ detect_prompt_type()           │ normalize_json_*()  │
│             │ split_prompt()                 │                     │
│             │ deduplicate_prompts()          │                     │
│             │                                │                     │
│             ▼                                ▼                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │   ContextNormalizer (Bidirectional Interface)          │        │
│  │   - normalize_input()                                  │        │
│  │   - normalize_output()                                 │        │
│  │   - normalize_ollama_output()                          │        │
│  │   - get_stats()                                        │        │
│  └────────────┬──────────────────────────────────────────┘        │
│               │                                                    │
│               │                                                    │
│               ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        Provider-Agnostic Processing Layer                  │ │
│  │                                                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │ │
│  │  │ OpenAI       │  │ Anthropic    │  │ Ollama       │     │ │
│  │  │ Parser       │  │ Parser       │  │ Parser       │     │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │ │
│  │                                                               │ │
│  │           UnifiedResponseParser                               │ │
│  │  (Auto-detects provider from response format)               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        Semantic Processing & Analysis Layer                 │ │
│  │                                                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │ │
│  │  │ Structured   │  │ Response     │  │ Streaming    │     │ │
│  │  │ Parser       │  │ Comparator   │  │ Normalizer   │     │ │
│  │  │              │  │              │  │              │     │ │
│  │  │ (JSON ↔ Text)│  │ (Similarity) │  │ (Real-time)  │     │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        Traditional NLP Processing Layer                      │ │
│  │                                                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │ │
│  │  │ Segmenter    │  │ Classifier   │  │ Deduplicator│     │ │
│  │  │              │  │              │  │              │     │ │
│  │  │ (Boundaries) │  │ (Code/Prose) │  │ (Hashing)    │     │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        Core Utilities Layer (text_utils.py)                 │ │
│  │                                                               │ │
│  │  segment_text()    hash_chunk()      extract_code_blocks() │ │
│  │  is_duplicate()    normalize_whitespace()                   │ │
│  │  count_tokens_estimate()                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Processing Pipeline

### Input Processing Pipeline

```
Raw Prompt
    ↓
[normalize_prompt]     ← Fix formatting, line endings, code fences
    ↓
[detect_prompt_type]   ← Identify: code, conversation, example, instruction, system
    ↓
[split_prompt]         ← Segment into chunks (with overlap for context)
    ↓
[deduplicate_prompts]  ← Mark duplicates using hashing
    ↓
[extract_code_blocks]  ← Identify and preserve code
    ↓
Enriched Segments
├─ id: prompt_123_seg_0
├─ type: instruction
├─ content: "..."
├─ hash: abc123...
├─ duplicate: false
├─ token_estimate: 250
├─ code_blocks: [...]
└─ timestamp: 2024-01-01T12:00:00
```

### Output Processing Pipeline

```
Raw LLM Response
    ↓
[Detect Provider Format]  ← OpenAI / Anthropic / Ollama / Generic
    ↓
[Provider Parser]         ← Extract content, usage, metadata
    ↓
[Remove Artifacts]        ← Strip <thinking>, <system>, model-specific markers
    ↓
[Normalize Code Blocks]   ← Fix fence formatting, language tags
    ↓
[Normalize Whitespace]    ← Collapse blank lines, trim
    ↓
[Extract Metadata]        ← Analyze structure (has_code, has_lists, etc.)
    ↓
Normalized Response
├─ normalized_content: "..."
├─ original_content: "..."
├─ provider: ollama
├─ model: llama2
├─ usage: {prompt_tokens: 10, completion_tokens: 5}
├─ metadata: {has_code: true, code_blocks: 2, ...}
├─ reduction_ratio: 0.05
├─ token_estimate: 250
└─ hash: xyz789...
```

## Class Hierarchy & Dependencies

```
PromptProcessor
├─ Uses: text_utils functions
│   ├─ segment_text()
│   ├─ hash_chunk()
│   ├─ normalize_whitespace()
│   └─ extract_code_blocks()
├─ normalize_prompt() [function]
├─ detect_prompt_type() [function]
├─ split_prompt() [function]
└─ deduplicate_prompts() [function]

ResponseNormalizer
├─ Uses: UnifiedResponseParser
├─ Uses: text_utils functions
├─ Core Methods:
│   ├─ normalize()
│   └─ normalize_json_response()
├─ Sub-classes:
│   ├─ StreamingNormalizer
│   ├─ ResponseComparator
│   └─ OllamaResponseHandler
└─ Helper Methods:
    ├─ _apply_model_quirks()
    ├─ _normalize_code_blocks()
    └─ _extract_metadata()

UnifiedResponseParser
├─ OpenAIParser
│   └─ parse() → Standard format
├─ AnthropicParser
│   └─ parse() → Standard format
├─ OllamaParser
│   └─ parse() → Standard format
├─ GenericParser
│   └─ parse() → Standard format
└─ _detect_provider() [auto-detection]

ContextNormalizer (FACADE)
├─ Wraps: PromptProcessor
├─ Wraps: ResponseNormalizer
├─ Wraps: OllamaResponseHandler
└─ Methods:
    ├─ normalize_input()
    ├─ normalize_output()
    └─ normalize_ollama_output()

MultiProviderNormalizer
├─ Wraps: ContextNormalizer
├─ provider_configs: {}
├─ register_provider()
├─ normalize_input_for_provider()
└─ normalize_outputs_from_providers()

StructuredParser
├─ parse()
├─ reconstruct()
├─ Fidelity Levels: MAXIMUM, HIGH, MEDIUM, LOW, MINIMAL
└─ Helper Methods:
    ├─ _extract_code_block()
    ├─ _extract_thinking_block()
    ├─ _parse_list()
    ├─ _parse_inline_formatting()
    └─ _apply_fidelity()

ContextSegmenter
├─ segment_text()
├─ segment_sentences()
├─ segment_paragraphs()
└─ Helper Methods:
    ├─ _extract_code_blocks()
    ├─ _split_sentences()
    └─ _classify_paragraph()

ContentClassifier
├─ classify() → ClassificationResult
├─ _score_code_syntax()
├─ _score_code_keywords()
├─ _score_indentation()
├─ _score_prose_patterns()
├─ _score_structured_data()
├─ _score_punctuation()
├─ _score_line_length()
└─ _score_whitespace()

ContentDeduplicator
├─ hash_text()
├─ is_duplicate()
├─ deduplicate_list()
├─ get_hash_info()
├─ get_statistics()
├─ save_cache()
└─ Helper Methods:
    ├─ _load_cache()
    ├─ _normalize_text()
    └─ _looks_like_code()
```

## Data Flow Examples

### Example 1: Simple Prompt Processing

```python
from hydracontext.core.prompt_processor import PromptProcessor

processor = PromptProcessor(max_chars=2048, overlap=200)
segments = processor.process("Your prompt here")

# Returns: [
#   {
#     "id": "prompt_0_seg_0",
#     "content": "Your prompt here",
#     "type": "instruction",
#     "hash": "...",
#     "duplicate": False,
#     "token_estimate": 4,
#     ...
#   }
# ]
```

### Example 2: Response Normalization with Provider Detection

```python
from hydracontext.core.response_processor import ResponseNormalizer

normalizer = ResponseNormalizer()

# Ollama response (auto-detected)
ollama_response = {
    "response": "Here's the answer...",
    "model": "llama2",
    "done": True,
    "eval_count": 100
}

normalized = normalizer.normalize_json_response(ollama_response)
# Returns: {
#   "normalized_content": "Here's the answer...",
#   "provider": "ollama",
#   "model": "llama2",
#   "usage": {"prompt_tokens": ?, "completion_tokens": 100, ...},
#   "metadata": {...},
#   ...
# }
```

### Example 3: Bidirectional Context Handling

```python
from hydracontext.core.bidirectional import ContextNormalizer

normalizer = ContextNormalizer()

# Normalize input
prompt_data = normalizer.normalize_input("Explain quantum computing")
print(prompt_data['content'])  # Clean, normalized prompt
print(prompt_data['type'])     # 'instruction'

# Send to LLM...
response = llm.generate(prompt_data['content'])

# Normalize output
response_data = normalizer.normalize_output(response, provider='ollama', model='llama2')
print(response_data['normalized_content'])  # Clean response
print(response_data['reduction_ratio'])     # How much was removed (artifacts, etc.)
```

### Example 4: Structural Analysis & Reconstruction

```python
from hydracontext.core.structured_parser import StructuredParser, FidelityLevel

parser = StructuredParser(fidelity=FidelityLevel.MAXIMUM)

# Parse text to JSON
structured = parser.parse("""
# Title

This is a paragraph with **bold** text.

```python
def hello():
    print("world")
```
""")

# Analyze structure
for block in structured['blocks']:
    print(f"{block['type']}: {block.get('content', '')[:50]}")

# Reconstruct at different fidelity
high_detail = parser.reconstruct(structured)
structured['fidelity'] = FidelityLevel.MINIMAL
low_detail = parser.reconstruct(structured)
```

## Provider Support Matrix

```
Provider    | Parser          | Auto-Detect | Token Tracking | Special Features
─────────────────────────────────────────────────────────────────────────────
OpenAI      | OpenAIParser    | Yes         | Full           | Message format
Anthropic   | AnthropicParser | Yes         | Full           | Content array
Ollama      | OllamaParser    | Yes         | Full           | Context window, streaming
Generic     | GenericParser   | Yes         | Partial        | Fallback support
─────────────────────────────────────────────────────────────────────────────
```

## Performance Characteristics

```
PromptProcessor:
  - normalize_prompt: O(n) where n = text length
  - detect_prompt_type: O(n) with regex matching
  - split_prompt: O(n) with overlap handling
  - deduplicate: O(1) per segment (hash lookup)
  
ResponseNormalizer:
  - normalize: O(n) with regex replacements
  - normalize_json_response: O(n) + provider parsing
  
ResponseComparator:
  - compare: O(m²) where m = number of responses
  - _compute_similarity: O(w) where w = unique words
  
ContentClassifier:
  - classify: O(n) with 8 feature scorers
  
ContentDeduplicator:
  - hash_text: O(n) with hashing
  - is_duplicate: O(1) with set lookup
  - deduplicate_list: O(k) where k = list length
```

## Integration Points

### With OpenAI

```python
from openai import OpenAI
from hydracontext.core.response_processor import ResponseNormalizer

client = OpenAI()
normalizer = ResponseNormalizer()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": normalized_prompt}]
)

normalized = normalizer.normalize_json_response(response.model_dump(), provider="openai")
```

### With Anthropic

```python
from anthropic import Anthropic
from hydracontext.core.response_processor import ResponseNormalizer

client = Anthropic()
normalizer = ResponseNormalizer()

response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": normalized_prompt}]
)

normalized = normalizer.normalize_json_response(response.model_dump(), provider="anthropic")
```

### With Ollama

```python
import requests
from hydracontext.core.response_processor import OllamaResponseHandler

handler = OllamaResponseHandler()

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",
        "prompt": normalized_prompt,
        "stream": False
    }
).json()

normalized = handler.normalize_ollama_response(response)
```

---

**Last Updated**: 2024-11-11  
**Framework Version**: See README.md for current version  
**Documentation**: See FEATURES_ANALYSIS.md for detailed API documentation
