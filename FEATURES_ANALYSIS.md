# HydraContext Comprehensive Feature Analysis

## Overview
HydraContext is a sophisticated context processing framework for LLMs that provides bidirectional normalization of prompts (input) and responses (output) across multiple providers. It combines modern AI processing features with traditional NLP techniques to enable consistent, efficient handling of context across different LLM providers (OpenAI, Anthropic, Ollama, etc.).

---

## 1. PROMPT PROCESSING FEATURES (hydracontext/core/prompt_processor.py)

### Main Class: `PromptProcessor`

**Purpose**: Stateful engine for processing prompts through a complete pipeline with caching and deduplication across multiple prompts.

### Key Functions

#### 1. `normalize_prompt(prompt: str) -> str`
**What it does**: Normalizes prompt formatting for consistent processing

**Capabilities**:
- Line ending harmonization (converts \r\n and \r to \n)
- Excess whitespace removal and leading/trailing strip
- Markdown code fence normalization (fixes ```python\n\n\ncode → ```python\ncode)
- Multiple blank line collapsing (max 2)
- List formatting normalization
- Heading formatting normalization

**Example**:
```python
raw = "# Title  \n\n\n```python\n\n\nprint('hello')\n```"
normalized = normalize_prompt(raw)
# Returns: "# Title\n\n```python\nprint('hello')\n```"
```

#### 2. `detect_prompt_type(text: str) -> str`
**What it does**: Classifies prompts by intent using heuristics

**Detected Types**:
- `"code"`: Contains code blocks (```), function defs, classes, or indented code patterns
- `"conversation"`: Multi-turn format with user:/assistant:/human:/ai: patterns
- `"example"`: Contains example patterns or input/output blocks
- `"instruction"`: Direct commands/questions (explain, describe, write, create, implement, fix, debug)
- `"system"`: System-level directives or metadata (### system: patterns)

**Example**:
```python
detect_prompt_type("def hello():\n    print('world')")  # → "code"
detect_prompt_type("User: Hello\nAssistant: Hi there")  # → "conversation"
detect_prompt_type("Explain how X works")  # → "instruction"
```

#### 3. `split_prompt(prompt: str, max_chars: int = 2048, overlap: int = 200) -> List[Dict]`
**What it does**: Splits prompts into logical segments for processing

**Features**:
- Respects max_chars limit per segment
- Applies overlap between segments for context preservation
- Returns segments with metadata (id, content, length, type)
- Delegates to `segment_text()` utility function

#### 4. `deduplicate_prompts(segments: List[Dict], seen_hashes: Optional[Set] = None) -> tuple`
**What it does**: Marks duplicate prompt segments using hash-based deduplication

**Features**:
- Hashes each segment content
- Tracks previously seen hashes
- Marks duplicates while returning ALL segments
- Maintains cumulative hash set across multiple prompt calls

**Returns**: (all_segments_with_duplicate_flags, updated_hash_set)

### Class Methods

#### `process(prompt: str, prompt_id: Optional[str] = None) -> List[Dict]`

**Complete Pipeline**:
1. Normalizes the prompt
2. Detects prompt type
3. Splits into segments
4. Deduplicates (marks duplicates)
5. Enriches with metadata

**Returned Segment Format**:
```python
{
    "id": "prompt_123_seg_0",
    "prompt_id": "prompt_123",
    "segment_index": 0,
    "type": "instruction",
    "normalized": True,
    "content": "...",
    "length": 245,
    "hash": "abc123...",
    "duplicate": False,
    "token_estimate": 61,
    "timestamp": "2024-01-01T12:00:00.123456",
    "code_blocks": [{"language": "python", "content": "..."}]  # if present
}
```

#### `process_batch(prompts: List[Union[str, Dict]]) -> List[List[Dict]]`
**Processes multiple prompts** in a single call. Accepts:
- List of strings
- List of dicts with `id` and `content` keys

#### `get_statistics() -> Dict`
**Returns**:
```python
{
    "processed_count": 5,
    "unique_hashes": 12,
    "max_chars": 2048,
    "overlap": 200
}
```

#### `reset()`
Clears hash cache and resets statistics

### File-Based Processing

#### `process_prompts(input_path, output_path, max_chars, overlap, format) -> Dict`

**Supports multiple input formats**:
- Plain text files (single prompt or separated by `---`)
- JSON files (list or dict with 'prompts' key)
- JSONL files (one prompt per line)

**Output formats**:
- JSON: Structured with metadata and statistics
- JSONL: One segment per line

### Key Capabilities

1. **Deterministic Processing**: Same input always produces same output
2. **Memory Caching**: Tracks seen hashes to identify repeated content
3. **Semantic Detection**: Understands prompt intent
4. **Code Preservation**: Extracts and preserves code blocks
5. **Batch Processing**: Handles multiple prompts efficiently
6. **Token Estimation**: Provides rough token counts (~4 chars = 1 token)
7. **Metadata Enrichment**: Adds context, timestamps, hashes to segments

### Use Cases

- **Context Window Management**: Break large prompts into manageable segments
- **Deduplication Caching**: Identify and skip repeated prompt content
- **Semantic Routing**: Route prompts to different models based on type
- **Prompt Optimization**: Normalize formatting for consistent processing
- **Multi-turn Conversations**: Process conversation history efficiently

---

## 2. BIDIRECTIONAL NORMALIZATION (hydracontext/core/bidirectional.py)

### Main Class: `ContextNormalizer`

**Purpose**: Unified interface for normalizing both prompts (input to LLM) and responses (output from LLM) across any provider.

### Core Methods

#### `normalize_input(prompt: str, prompt_id: Optional[str] = None) -> Dict`

**What it does**: Normalizes a prompt before sending to LLM

**Returns**:
```python
{
    'content': 'Normalized prompt text',
    'type': 'instruction',  # From detect_prompt_type
    'token_estimate': 250,
    'segments': [segment1, segment2, ...],  # From PromptProcessor
    'direction': 'input'
}
```

**Smart Behavior**:
- Single-segment prompts: Returns simplified format
- Multi-segment prompts: Rejoins segments with '\n\n'

#### `normalize_output(response: str, provider: Optional[str], model: Optional[str]) -> Dict`

**What it does**: Normalizes response from any LLM provider

**Returns**:
```python
{
    'normalized_content': 'Cleaned response',
    'original_content': 'Raw response',
    'provider': 'ollama',
    'model': 'llama2',
    'original_length': 1500,
    'normalized_length': 1450,
    'reduction_ratio': 0.033,
    'token_estimate': 362,
    'metadata': {
        'has_code': True,
        'code_blocks': 2,
        'has_lists': True,
        'has_headings': False,
        'line_count': 45,
        'paragraph_count': 8
    },
    'hash': 'xyz789...',
    'timestamp': '2024-01-01T12:00:00.123456',
    'direction': 'output'
}
```

#### `normalize_ollama_output(response_dict: Dict) -> Dict`

**Specialized for Ollama JSON responses**:
- Extracts content from Ollama format
- Preserves Ollama-specific metadata
- Handles both response and message.content formats

#### `get_stats() -> Dict`

**Returns**: Processing statistics from underlying PromptProcessor

### Secondary Class: `MultiProviderNormalizer`

**Purpose**: Normalize context across multiple LLM providers simultaneously

#### Key Methods

- `register_provider(name: str, config: Optional[Dict])`: Register provider-specific config
- `normalize_input_for_provider(prompt: str, provider: str)`: Tailored normalization for specific provider
- `normalize_outputs_from_providers(responses: List[Dict])`: Normalize multiple provider responses

**Provider-Specific Optimizations**:
- **Ollama**: Optimized for longer contexts
- **OpenAI**: Prefers structured prompts
- **Anthropic**: Prefers natural language

### Key Capabilities

1. **Single Normalizer Interface**: One class handles both input and output
2. **Provider Agnostic**: Works with any LLM provider
3. **Stateful Processing**: Maintains context across calls
4. **Metadata Rich**: Captures detailed information about processing
5. **Multi-provider Support**: Can process outputs from different providers

### Use Cases

- **Provider Agnostic Applications**: Build systems that work with multiple LLMs
- **Response Comparison**: Compare responses from different models
- **Consistent API**: Unified interface for prompt and response handling
- **Provider Migration**: Switch between providers without code changes
- **Multi-model Ensemble**: Process responses from multiple models

---

## 3. RESPONSE PROCESSING (hydracontext/core/response_processor.py)

### Main Class: `ResponseNormalizer`

**Purpose**: Normalize responses from different LLM providers into consistent format, handling model-specific quirks and formatting inconsistencies.

### Configuration

```python
ResponseNormalizer(
    remove_thinking=True,              # Remove <thinking> tags
    remove_system=True,                # Remove <system> messages
    normalize_code_blocks=True,        # Fix code fence formatting
    strip_artifacts=True               # Remove model-specific artifacts
)
```

### Core Method: `normalize(response: str, provider: Optional[str], model: Optional[str]) -> Dict`

**What it does**: Complete normalization pipeline for any response

**Pipeline**:
1. Removes thinking blocks (if enabled)
2. Removes system artifacts (if enabled)
3. Applies model-specific cleaning
4. Normalizes whitespace
5. Normalizes code blocks
6. Extracts metadata

**Returns**:
```python
{
    'normalized_content': 'Clean response text',
    'original_content': 'Original response',
    'provider': 'ollama',
    'model': 'llama2',
    'original_length': 2000,
    'normalized_length': 1900,
    'reduction_ratio': 0.05,
    'token_estimate': 475,
    'metadata': {
        'has_code': True,
        'code_blocks': 3,
        'has_lists': False,
        'has_headings': True,
        'line_count': 67,
        'paragraph_count': 12
    },
    'hash': 'abc123...',
    'timestamp': '2024-01-01T12:00:00.123456'
}
```

### Advanced Method: `normalize_json_response(raw_response: Union[Dict, str], provider: Optional[str]) -> Dict`

**Purpose**: Normalize provider-specific JSON API responses

**Handles**:
- JSON string or dict input
- Delegates to provider-specific parser
- Applies content normalization
- Merges parsed metadata with normalized content

**Example**:
```python
# OpenAI response
openai_resp = {
    "choices": [{"message": {"content": "Hello!"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
}
normalized = normalizer.normalize_json_response(openai_resp, "openai")

# Ollama response
ollama_resp = {"response": "Hello!", "model": "llama2", "done": True}
normalized = normalizer.normalize_json_response(ollama_resp, "ollama")
```

### Pattern Removal Features

**Thinking Patterns** (regex):
```
<thinking>.*?</thinking>
<thought>.*?</thought>
[THINKING]...[/THINKING]
Let me think...
```

**System Patterns** (regex):
```
<system>.*?</system>
[SYSTEM]...[/SYSTEM]
System:...
```

**Model-Specific Quirks**:
- **llama2**: Removes [INST]...[/INST] wrappers and </s> tokens
- **mistral**: Removes <|im_start|>...<|im_end|> markers
- **codellama**: Removes ```output``` blocks

### Code Block Normalization

- Fixes inconsistent fence formatting
- Ensures consistent spacing after language tags
- Converts language tags to lowercase

### Helper Methods

- `_apply_model_quirks(text, model)`: Apply model-specific patterns
- `_normalize_code_blocks(text)`: Fix code fence formatting
- `_extract_metadata(text)`: Extract structural info

### Secondary Class: `StreamingNormalizer`

**Purpose**: Normalize streaming responses chunk-by-chunk (especially for Ollama)

**Methods**:
- `process_chunk(chunk: str)`: Process single chunk, returns normalized if complete unit found
- `flush()`: Flush remaining buffered content

**Detects Complete Units**:
- Sentence endings (., !, ?)
- Complete code blocks (matching ```)
- Paragraph breaks (\n\n)

**Use Case**: Real-time normalization of streaming LLM responses

### Secondary Class: `ResponseComparator`

**Purpose**: Compare responses from different models/providers

#### Method: `compare(responses: List[Dict], similarity_threshold: float = 0.8) -> Dict`

**Returns**:
```python
{
    'total_responses': 3,
    'unique_responses': 2,
    'duplicate_responses': 1,
    'pairwise_similarities': [
        {'pair': (0, 1), 'models': ('model_a', 'model_b'), 'similarity': 0.85, 'similar': True}
    ],
    'average_similarity': 0.82,
    'providers': ['ollama', 'openai'],
    'models': ['llama2', 'gpt-4'],
    'length_stats': {'min': 1200, 'max': 1500, 'avg': 1350},
    'token_stats': {'min': 300, 'max': 375, 'avg': 337}
}
```

**Features**:
- Pairwise similarity computation
- Duplicate detection using hashes
- Length and token statistics
- Provider/model tracking

### Secondary Class: `OllamaResponseHandler`

**Purpose**: Specialized handling for Ollama-specific features

#### Methods

- `normalize_ollama_response(response_dict)`: Normalize Ollama API response with Ollama-specific metadata
- `process_streaming_response(stream)`: Process Ollama streaming responses

**Ollama Metadata Captured**:
```python
'ollama_metadata': {
    'model': 'llama2',
    'done': True,
    'total_duration': 5000000000,
    'load_duration': 1000000,
    'prompt_eval_count': 26,
    'eval_count': 282,
    'context': [1, 2, 3, ...]  # Context window for multi-turn
}
```

### Key Capabilities

1. **Multi-provider Support**: Handles OpenAI, Anthropic, Ollama, generic
2. **Artifact Removal**: Cleans thinking blocks, system messages, model-specific markers
3. **Streaming Support**: Process streaming responses in real-time
4. **Comparison**: Compare responses from multiple models
5. **Metadata Extraction**: Capture structural information
6. **Flexible Configuration**: Customize what gets removed/normalized

### Use Cases

- **Response Cleanup**: Remove LLM artifacts for cleaner output
- **Provider Switching**: Normalize responses from different providers
- **Streaming Processing**: Handle real-time LLM output
- **Quality Assessment**: Compare model outputs
- **Output Deduplication**: Identify similar responses
- **Ollama Integration**: Special handling for local LLMs

---

## 4. STRUCTURED PARSING (hydracontext/core/structured_parser.py)

### Main Class: `StructuredParser`

**Purpose**: Parse unstructured text into structured JSON representation that preserves all information while enabling semantic operations.

### Content Types (Enum)

```python
HEADING, PARAGRAPH, CODE_BLOCK, LIST, TABLE, QUOTE, LINK, IMAGE,
INLINE_CODE, BOLD, ITALIC, THINKING, INSTRUCTION, QUESTION, ANSWER,
EXAMPLE, METADATA
```

### Fidelity Levels

```python
MAXIMUM    # Keep everything, all structure
HIGH       # Keep semantic structure, normalize formatting
MEDIUM     # Keep main content, discard auxiliary
LOW        # Summary only, discard details
MINIMAL    # Core message only
```

### Core Method: `parse(text: str, metadata: Optional[Dict]) -> Dict`

**What it does**: Converts unstructured text into hierarchical JSON structure

**Returns**:
```python
{
    'version': '1.0',
    'fidelity': 'maximum',
    'metadata': {...},
    'blocks': [
        {
            'type': 'heading',
            'level': 1,
            'content': 'Title',
            'inline_formatting': {
                'has_bold': False,
                'has_italic': False,
                'has_inline_code': False,
                'has_links': False
            }
        },
        {
            'type': 'paragraph',
            'content': 'This is a paragraph with **bold** text.',
            'inline_formatting': {'has_bold': True, ...},
            'length': 45,
            'line_count': 1
        },
        {
            'type': 'code_block',
            'language': 'python',
            'content': 'def hello():\n    print("world")',
            'length': 34,
            'line_count': 2
        },
        {
            'type': 'list',
            'list_type': 'unordered',
            'items': ['Item 1', 'Item 2', 'Item 3'],
            'item_count': 3
        },
        {
            'type': 'question',
            'content': 'How does this work?',
            'inline_formatting': {...}
        },
        {
            'type': 'thinking',
            'content': 'Internal reasoning...',
            'removable': True  # Can be removed at lower fidelity
        }
    ],
    'statistics': {
        'total_blocks': 6,
        'block_types': {
            'heading': 1,
            'paragraph': 1,
            'code_block': 1,
            'list': 1,
            'question': 1,
            'thinking': 1,
            ...
        },
        'total_length': 200,
        'has_code': True,
        'has_lists': True,
        'has_headings': True
    }
}
```

### Block Detection

**Headings**: Markdown style (# - ######)
**Code Blocks**: Fenced (``` or ~~~)
**Lists**: Bullet (-, *, +) or ordered (1., 2., etc.)
**Thinking Blocks**: <thinking>...</thinking> tags
**Questions**: Lines ending with ?
**Paragraphs**: Regular text blocks

### Inline Formatting Detection

Detects within blocks:
- **Bold**: **text**
- **Italic**: *text*
- **Inline Code**: `code`
- **Links**: [text](url)

### Reconstruction Method: `reconstruct(structured: Dict) -> str`

**What it does**: Converts structured JSON back to text

**Features**:
- Rebuilds markdown formatting
- Respects fidelity levels (removes thinking blocks at low fidelity)
- Preserves structure
- Lossless round-trip conversion

### Helper Methods

- `_extract_code_block()`: Extracts code blocks with language tags
- `_extract_thinking_block()`: Extracts reasoning blocks
- `_parse_list()`: Parses ordered and unordered lists
- `_parse_inline_formatting()`: Detects formatting markers
- `_apply_fidelity()`: Filters blocks based on fidelity level
- `_compute_statistics()`: Calculates block statistics

### Convenience Functions

```python
# Parse to JSON
parsed = parse_to_json(text, fidelity=FidelityLevel.MAXIMUM)

# Reconstruct from JSON
text = json_to_text(parsed_json)
```

### Key Capabilities

1. **Lossless Conversion**: Text ↔ JSON without information loss
2. **Semantic Structure**: Hierarchical understanding of content
3. **Fidelity Control**: Choose detail level for different purposes
4. **Metadata Extraction**: Capture formatting and structure
5. **Inline Analysis**: Detect formatted text within blocks
6. **Statistics**: Analyze document structure

### Use Cases

- **Information Preservation**: Keep all data while changing formats
- **Selective Compression**: Remove unnecessary detail while keeping essential info
- **Multi-format Processing**: Convert between text and structured formats
- **Content Analysis**: Understand document structure
- **Response Fusion**: Combine multiple responses by merging blocks
- **Context Compression**: Reduce context size while controlling fidelity

---

## 5. PROVIDER PARSERS (hydracontext/core/provider_parsers.py)

### Standard Response Format

All providers normalize to this unified format:

```python
{
    "content": str,           # The actual response text
    "provider": str,          # Provider name (openai, anthropic, ollama, generic)
    "model": str,             # Model name
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    },
    "finish_reason": str,     # Why generation stopped (stop, length, error, etc.)
    "metadata": dict,         # Provider-specific metadata
    "timestamp": str          # ISO format timestamp
}
```

### Provider-Specific Parsers

#### 1. `OpenAIParser`

**Input Format**:
```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello!"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}
```

**Extraction**:
- Content from `choices[0].message.content`
- Usage already in standard format
- Captures id, created, object metadata

#### 2. `AnthropicParser`

**Input Format**:
```python
{
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "type": "message",
    "content": [
        {"type": "text", "text": "Hello!"}
    ],
    "model": "claude-3-opus-20240229",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 25
    }
}
```

**Extraction**:
- Content from content array (text items)
- Maps input_tokens → prompt_tokens, output_tokens → completion_tokens
- Captures id, type, stop_sequence metadata

#### 3. `OllamaParser`

**Input Format**:
```python
{
    "model": "llama2",
    "created_at": "2023-08-04T19:22:45.499127Z",
    "response": "Hello!",
    "done": true,
    "context": [1, 2, 3, ...],
    "total_duration": 4883583458,
    "load_duration": 1334875,
    "prompt_eval_count": 26,
    "eval_count": 282,
    "prompt_eval_duration": 342546000,
    "eval_duration": 4535599000
}
```

**Extraction**:
- Content from `response` field
- Calculates tokens from eval_count fields
- Captures all performance metrics
- Preserves context array (important for multi-turn)

**Special Handling**: Also supports `message.content` format for chat endpoints

#### 4. `GenericParser`

**Fallback for unknown providers**

**Strategies**:
- Checks for common content fields: content, response, text, message, output
- Falls back to string conversion
- Captures raw dict as metadata

### Main Class: `UnifiedResponseParser`

**Purpose**: Routes to provider-specific parsers, auto-detects provider format

#### Method: `parse(raw_response: Any, provider: Optional[str] = None) -> Dict`

**With Provider Specified**:
```python
parser = UnifiedResponseParser()
parsed = parser.parse(openai_response, provider="openai")
```

**Auto-Detection**:
```python
# Provider auto-detected from response structure
parsed = parser.parse(response_dict)
```

#### Auto-Detection Logic

```
if 'choices' in response AND 'usage' in response:
    → OpenAI

if 'content' is list AND has text items:
    → Anthropic

if 'response' in response AND 'done' in response:
    → Ollama

if 'message' in response AND 'model' in response:
    → Ollama (chat format)

else:
    → Generic
```

### Key Capabilities

1. **Provider Agnostic**: Single interface for all providers
2. **Auto-Detection**: Intelligently detects provider format
3. **Usage Normalization**: Maps provider-specific token counts to standard format
4. **Metadata Preservation**: Captures provider-specific data
5. **Fallback Support**: Handles unknown providers gracefully
6. **Token Tracking**: Normalizes all token usage formats

### Use Cases

- **Multi-provider Applications**: Support multiple LLM providers
- **Provider Abstraction**: Hide provider details from business logic
- **Response Comparison**: Compare responses in unified format
- **Analytics**: Consistent token usage tracking
- **Provider Migration**: Switch providers without code changes

---

## 6. TRADITIONAL FEATURES

### 6.1 CONTEXT SEGMENTATION (segmenter.py)

#### Main Class: `ContextSegmenter`

**Purpose**: Intelligent text segmentation with support for prose and code

#### Configuration

```python
ContextSegmenter(
    min_sentence_length=3,      # Minimum chars for valid sentence
    preserve_code=True          # Keep code blocks intact
)
```

#### Segment Types

```python
SENTENCE, PARAGRAPH, CODE_BLOCK, LIST_ITEM, HEADING
```

#### Core Method: `segment_text(text: str, granularity: str = 'sentence') -> List[Segment]`

**Granularities**:
- `'sentence'`: Split by sentence boundaries
- `'paragraph'`: Split by paragraph breaks

#### Detailed Methods

**`segment_sentences(text: str)`**:
- Preserves code blocks intact
- Detects sentence boundaries with abbreviation handling
- Handles quoted text exceptions
- Returns Segment objects with position tracking

**`segment_paragraphs(text: str)`**:
- Splits on double newlines
- Classifies paragraph type (heading, list, code, paragraph)
- Preserves position information

#### Segment Object

```python
@dataclass
class Segment:
    text: str                    # Segment content
    type: SegmentType            # Sentence, paragraph, code, etc.
    start_pos: int               # Start position in original text
    end_pos: int                 # End position in original text
    metadata: Optional[Dict]     # Additional metadata
```

#### Code Block Detection

Patterns supported:
- Fenced blocks: ``` ... ```
- Alternative fenced: ~~~ ... ~~~
- Indented code: 4 spaces or tab indentation

#### Abbreviation Handling

Knows common abbreviations that don't end sentences:
- Titles: Dr, Mr, Mrs, Ms, Prof, Sr, Jr
- Academic: vs, etc, e.g, i.e, cf, al
- Business: vol, fig, approx, appt, dept

#### Key Capabilities

1. **Smart Boundaries**: Understands sentence vs. abbreviation
2. **Code Preservation**: Protects code blocks from splitting
3. **Position Tracking**: Know exact location of segments
4. **Type Classification**: Identifies segment purpose
5. **Abbreviation Aware**: Handles domain-specific abbreviations

---

### 6.2 CONTENT CLASSIFICATION (classifier.py)

#### Main Class: `ContentClassifier`

**Purpose**: Heuristic-based classifier distinguishing code from prose

#### Content Types

```python
PROSE, CODE, STRUCTURED_DATA, MIXED, UNKNOWN
```

#### Configuration

```python
ContentClassifier(threshold=0.6)  # Confidence threshold (0.0-1.0)
```

#### Core Method: `classify(text: str) -> ClassificationResult`

**Returns**:
```python
ClassificationResult(
    content_type=ContentType.CODE,
    confidence=0.85,  # 0.0 to 1.0
    indicators={
        'code_syntax': 0.8,
        'code_keywords': 0.75,
        'indentation': 0.9,
        'prose_patterns': 0.2,
        'structured_data': 0.0,
        'punctuation': 0.15,
        'line_length': 0.3,
        'whitespace': 0.7
    },
    metadata={
        'code_score': 0.815,
        'prose_score': 0.183,
        'structured_score': 0.0,
        'char_count': 350,
        'line_count': 12
    }
)
```

#### Classification Scoring Mechanism

**Code Score** (weighted):
- Code syntax patterns: 0.3
- Code keywords: 0.25
- Indentation: 0.2
- Inverse prose patterns: 0.25

**Prose Score** (weighted):
- Prose patterns: 0.4
- Punctuation variety: 0.3
- Inverse code syntax: 0.3

**Structured Score**:
- JSON/XML/YAML pattern matching: direct score

**Decision Logic**:
```
if structured_score > 0.7:
    → STRUCTURED_DATA
elif code_score > prose_score AND code_score >= threshold:
    → CODE
elif prose_score >= threshold:
    → PROSE
elif |code_score - prose_score| < 0.2:
    → MIXED
else:
    → UNKNOWN
```

#### Feature Scorers

**Code Syntax** (patterns + bracket density):
- Brackets: { } [ ] ( ) < >
- Operators: = < > ! + - * / etc.
- Arrow functions: => ->
- Logical operators: && ||
- Import statements
- Access modifiers

**Code Keywords**:
- function, class, def, return, if, else, for, while
- import, from, const, let, var
- int, float, string, void, async, await
- try, catch, throw, new, this, self, lambda, yield

**Indentation Pattern**:
- Measures leading whitespace consistency
- Checks for multiples of 2 or 4 (standard indent sizes)
- Bonus for consistent indentation

**Prose Patterns**:
- Articles/prepositions: the, a, an, and, or, but, in, on, at, to, for, of, with
- Verb forms: is, are, was, were, be, been, being
- Sentence boundaries: [.!?] followed by capital letter

**Structured Data**:
- JSON validation: {…} or […]
- XML detection: <tag>…</tag>
- YAML patterns: key: value format

**Punctuation Variety**:
- Counts . , ? ! : ; usage
- Measures sentence-ending punctuation density

**Line Length**:
- Prose: 60-100 chars average (0.8)
- Code: 30-60 chars average (0.3)

**Whitespace Ratio**:
- Code has more whitespace
- Measured as spaces/tabs/newlines percentage
- Includes blank line ratio

#### Key Capabilities

1. **Multi-feature Analysis**: 8 independent feature scorers
2. **Confidence Scores**: Know how certain the classification is
3. **Mixed Detection**: Recognize hybrid content
4. **Detailed Indicators**: See which features contributed to decision
5. **Customizable Threshold**: Tune sensitivity

---

### 6.3 CONTENT DEDUPLICATION (deduplicator.py)

#### Main Class: `ContentDeduplicator`

**Purpose**: Hash-based content deduplication with persistent caching

#### Configuration

```python
ContentDeduplicator(
    algorithm='sha256',           # 'md5', 'sha256', 'blake2b'
    normalize=True,               # Normalize before hashing
    cache_path=Path('cache.jsonl'),  # Persistent cache file
    min_length=10                 # Minimum text length to process
)
```

#### Data Structure

```python
@dataclass
class ContentHash:
    hash: str                     # Hex digest
    text: str                     # Content preview (first 200 chars)
    first_seen: str               # ISO timestamp
    occurrences: int = 1          # How many times seen
    metadata: Optional[Dict] = None
```

#### Core Methods

**`hash_text(text: str) -> str`**:
- Generates hash using configured algorithm
- Optionally normalizes text before hashing
- Returns hex digest

**`is_duplicate(text: str, record: bool = True) -> bool`**:
- Checks if text has been seen before
- Can optionally record new content
- Updates occurrence counts
- Returns: True if duplicate, False if unique

**`deduplicate_list(texts: List[str]) -> List[str]`**:
- Removes duplicates from list
- Preserves order
- Updates global tracking
- Checks both global cache and current batch

**`get_hash_info(text: str) -> Optional[ContentHash]`**:
- Retrieve metadata for a text
- Includes first-seen timestamp
- Shows occurrence count

**`get_statistics() -> Dict`**:
```python
{
    'total_processed': 1000,
    'unique_content': 850,
    'duplicates_found': 150,
    'cache_hits': 45,
    'unique_hashes': 850,
    'dedup_ratio': 0.15  # 15% duplicates
}
```

#### Persistent Caching

**Format**: JSONL (one hash per line)

**Methods**:
- `save_cache(path)`: Write cache to disk
- `_load_cache()`: Load cache on initialization
- Auto-loads if cache_path exists

**Example**:
```python
dedup = ContentDeduplicator(cache_path=Path('hashes.jsonl'))
# Auto-loads existing cache from hashes.jsonl
```

#### Text Normalization

**Aggressive Normalization** (for prose):
- Convert to lowercase
- Collapse whitespace
- Strip punctuation

**Conservative Normalization** (for code):
- Lowercase conversion only
- Preserves structure

**Detection**: Uses heuristics (brackets, keywords) to distinguish

#### Hash Algorithms

- **MD5**: Fast, 128-bit (legacy)
- **SHA256**: Secure, 256-bit (default)
- **BLAKE2b**: Fast & secure, 512-bit

#### Key Capabilities

1. **Multiple Algorithms**: Choose speed vs security
2. **Persistent Storage**: Cache survives restarts
3. **Batch Processing**: Deduplicate lists efficiently
4. **Content-aware Normalization**: Handles code differently
5. **Statistics Tracking**: Monitor deduplication effectiveness
6. **Export**: Save hashes to JSONL or CSV

### Use Cases

- **Prompt Caching**: Skip processing repeated prompts
- **Data Deduplication**: Remove duplicate training data
- **Cache Management**: Identify which content to skip
- **Statistics**: Measure content redundancy

---

## 7. TEXT UTILITIES (text_utils.py)

Helper functions used across all modules:

#### `segment_text(text: str, max_chars: int, overlap: int) -> List[Dict]`
- Chunks text with optional overlap
- Returns list of segment dicts

#### `hash_chunk(text: str, algorithm: str) -> str`
- Generate cryptographic hash
- Supports md5, sha1, sha256

#### `is_duplicate(hash_value: str, seen_hashes: Set) -> bool`
- Check if hash is in set
- Simple membership test

#### `normalize_whitespace(text: str) -> str`
- Normalize line endings
- Remove trailing whitespace
- Collapse blank lines to max 2

#### `extract_code_blocks(text: str) -> List[Dict]`
- Extract fenced code blocks
- Returns language and content
- Includes positions in original text

#### `count_tokens_estimate(text: str) -> int`
- Rough token count (~4 chars ≈ 1 token)
- Avoids dependency on tiktoken
- Good for planning

---

## INTEGRATION PATTERNS

### Pattern 1: Simple Input/Output Normalization

```python
from hydracontext.core.bidirectional import ContextNormalizer

normalizer = ContextNormalizer()

# Normalize input
prompt_data = normalizer.normalize_input("Explain recursion")
clean_prompt = prompt_data['content']

# Send to LLM and get response...
response = llm.generate(clean_prompt)

# Normalize output
response_data = normalizer.normalize_output(response, provider='ollama', model='llama2')
clean_response = response_data['normalized_content']
```

### Pattern 2: Multi-Model Comparison

```python
from hydracontext.core.response_processor import ResponseComparator

comparator = ResponseComparator()

responses = [
    {'normalized_content': 'Response 1...', 'model': 'gpt-4'},
    {'normalized_content': 'Response 2...', 'model': 'llama2'},
    {'normalized_content': 'Response 3...', 'model': 'claude-3'}
]

comparison = comparator.compare(responses, similarity_threshold=0.8)
# Identifies duplicates, similarities, stats
```

### Pattern 3: Streaming Response Processing

```python
from hydracontext.core.response_processor import OllamaResponseHandler

handler = OllamaResponseHandler()

for chunk in ollama_streaming_response:
    normalized = handler.process_streaming_response(chunk)
    for item in normalized:
        print(item['content'])
```

### Pattern 4: Structured Content Analysis

```python
from hydracontext.core.structured_parser import StructuredParser, FidelityLevel

parser = StructuredParser(fidelity=FidelityLevel.HIGH)

# Parse response
structured = parser.parse(response_text)

# Analyze blocks
for block in structured['blocks']:
    if block['type'] == 'code_block':
        print(f"Found {block['language']} code")

# Reconstruct at different fidelity
low_detail = parser.reconstruct(structured)
```

### Pattern 5: Batch Prompt Processing

```python
from hydracontext.core.prompt_processor import PromptProcessor

processor = PromptProcessor(max_chars=2048, overlap=200)

prompts = [
    "First prompt...",
    "Second prompt...",
    {"id": "custom_id", "content": "Third prompt..."}
]

results = processor.process_batch(prompts)
stats = processor.get_statistics()
```

### Pattern 6: Multi-Provider Setup

```python
from hydracontext.core.bidirectional import MultiProviderNormalizer
from hydracontext.core.provider_parsers import UnifiedResponseParser

normalizer = MultiProviderNormalizer()
normalizer.register_provider('ollama', {'max_context': 4096})
normalizer.register_provider('openai', {'max_context': 8000})

# Tailor prompt for each provider
ollama_prompt = normalizer.normalize_input_for_provider(prompt, 'ollama')
openai_prompt = normalizer.normalize_input_for_provider(prompt, 'openai')

# Normalize responses from different providers
responses = [
    {'provider': 'ollama', 'model': 'llama2', 'response': resp1},
    {'provider': 'openai', 'model': 'gpt-4', 'response': resp2}
]

normalized = normalizer.normalize_outputs_from_providers(responses)
```

---

## KEY ARCHITECTURAL INSIGHTS

1. **Layered Design**: From low-level text utilities → segmentation → classification → parsing → normalization → unified interface

2. **Separation of Concerns**:
   - Input processing (PromptProcessor)
   - Output processing (ResponseNormalizer)
   - Unified interface (ContextNormalizer)
   - Provider abstraction (UnifiedResponseParser)

3. **Provider Agnostic**: Works seamlessly with OpenAI, Anthropic, Ollama, and custom providers

4. **Stateful Processing**: Maintains caches across calls for efficiency and deduplication

5. **Metadata Rich**: Every operation captures detailed metadata for analysis and debugging

6. **Configurable Fidelity**: Choose level of detail preservation for different use cases

7. **Extensible**: Easy to add new providers, content types, or processing stages

8. **Lossless Conversion**: Can convert between formats without losing information

