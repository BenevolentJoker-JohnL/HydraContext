# HydraContext Documentation Index

This directory contains comprehensive documentation for the HydraContext framework.

## Quick Navigation

| Document | Purpose | Audience | Length |
|----------|---------|----------|--------|
| **QUICK_REFERENCE.md** | API cheat sheet, common patterns, code examples | Developers actively coding | 592 lines |
| **FEATURES_ANALYSIS.md** | Deep dive into every feature and capability | Architects, lead developers | 1,267 lines |
| **ARCHITECTURE_DIAGRAM.md** | System design, data flows, integration points | System architects, integrators | 435 lines |
| **README.md** | Overview, installation, getting started | Everyone | See main repo |

---

## Document Summaries

### QUICK_REFERENCE.md
**Best for**: Looking up API calls, copying code examples, troubleshooting

Contains:
- Installation and imports
- Core API for all 8 major components
- Common patterns (OpenAI, Anthropic, Ollama, multi-model)
- Configuration reference
- Troubleshooting table
- FAQs

**Example sections**:
```python
normalizer = ContextNormalizer()
prompt_data = normalizer.normalize_input("...")
response_data = normalizer.normalize_output("...")
```

**Start here if**: You want to code now.

---

### FEATURES_ANALYSIS.md
**Best for**: Understanding what HydraContext can do, detailed capability reference

Contains deep analysis of:

1. **Prompt Processing** (PromptProcessor)
   - normalize_prompt() - formatting normalization
   - detect_prompt_type() - classification
   - split_prompt() - segmentation
   - deduplicate_prompts() - duplicate detection
   - process() - full pipeline
   - process_batch() - batch operations

2. **Bidirectional Normalization** (ContextNormalizer)
   - normalize_input() - prompt preprocessing
   - normalize_output() - response postprocessing
   - normalize_ollama_output() - Ollama-specific
   - MultiProviderNormalizer - multi-provider support

3. **Response Processing** (ResponseNormalizer)
   - normalize() - text cleanup
   - normalize_json_response() - API response parsing
   - StreamingNormalizer - real-time processing
   - ResponseComparator - multi-response analysis
   - OllamaResponseHandler - Ollama streaming

4. **Structured Parsing** (StructuredParser)
   - parse() - text to JSON conversion
   - reconstruct() - JSON to text conversion
   - Fidelity levels - MAXIMUM to MINIMAL
   - Block detection - headings, code, lists, etc.

5. **Provider Parsers** (UnifiedResponseParser)
   - OpenAI parser
   - Anthropic parser
   - Ollama parser
   - Generic parser
   - Auto-detection logic

6. **Traditional Features**
   - ContextSegmenter - sentence/paragraph boundaries
   - ContentClassifier - code/prose/structured detection
   - ContentDeduplicator - hash-based deduplication

**Each section includes**:
- Main classes/functions
- Parameters and configuration
- Return formats with examples
- Key capabilities
- Use cases
- Algorithms and scoring mechanisms

**Start here if**: You want to understand the framework deeply.

---

### ARCHITECTURE_DIAGRAM.md
**Best for**: Understanding system design, integrating with external systems

Contains:

1. **System Overview Diagram**
   - Component relationships
   - Data flow from input to output
   - Layer organization

2. **Feature Processing Pipelines**
   - Input processing pipeline (5 stages)
   - Output processing pipeline (6 stages)
   - Data structures at each stage

3. **Class Hierarchy & Dependencies**
   - Full dependency graph
   - Which classes use which
   - Composition relationships

4. **Data Flow Examples**
   - Simple prompt processing
   - Response normalization with auto-detection
   - Bidirectional context handling
   - Structural analysis and reconstruction

5. **Provider Support Matrix**
   - Which providers are supported
   - Detection capabilities
   - Token tracking
   - Special features per provider

6. **Performance Characteristics**
   - Time complexity for each operation
   - Space complexity considerations

7. **Integration Points**
   - Code examples for OpenAI integration
   - Code examples for Anthropic integration
   - Code examples for Ollama integration

**Start here if**: You're integrating with external systems or designing the overall solution.

---

## How to Use This Documentation

### Scenario 1: "I want to normalize a prompt and response"
1. Go to QUICK_REFERENCE.md
2. Find "Bidirectional Normalization" section
3. Copy the code example
4. Done in 30 seconds

### Scenario 2: "I need to understand how multi-provider support works"
1. Go to FEATURES_ANALYSIS.md
2. Find "Provider Parsers" section
3. Read about UnifiedResponseParser and auto-detection
4. Find "MultiProviderNormalizer" in Bidirectional section
5. Reference ARCHITECTURE_DIAGRAM.md "Provider Support Matrix"

### Scenario 3: "I'm integrating HydraContext with our system"
1. Read ARCHITECTURE_DIAGRAM.md overview
2. Check "Integration Points" section for your provider
3. Look at FEATURES_ANALYSIS.md for detailed options
4. Use QUICK_REFERENCE.md for code snippets

### Scenario 4: "I need to troubleshoot something"
1. Check QUICK_REFERENCE.md "Troubleshooting" table
2. Look at relevant section in FEATURES_ANALYSIS.md for deep dive
3. Check configuration options
4. Review code examples in ARCHITECTURE_DIAGRAM.md

---

## Feature Map

### By Use Case

**Context Window Management**
- PromptProcessor (split_prompt)
- Text segmentation with overlap
- See FEATURES_ANALYSIS.md section 1

**Multi-LLM Support**
- UnifiedResponseParser (auto-detection)
- ContextNormalizer (bidirectional)
- MultiProviderNormalizer
- See FEATURES_ANALYSIS.md sections 2 & 5

**Response Cleanup**
- ResponseNormalizer (artifact removal)
- Pattern removal (thinking, system blocks)
- Code block normalization
- See FEATURES_ANALYSIS.md section 3

**Content Analysis**
- StructuredParser (text to JSON)
- ContentClassifier (code/prose detection)
- ContentSegmenter (boundary detection)
- See FEATURES_ANALYSIS.md sections 4 & 6

**Deduplication & Caching**
- ContentDeduplicator (hash-based)
- PromptProcessor (duplicate marking)
- Persistent JSONL cache
- See FEATURES_ANALYSIS.md section 6.3

**Streaming & Real-time**
- StreamingNormalizer (chunk processing)
- OllamaResponseHandler (streaming)
- process_chunk() + flush() pattern
- See QUICK_REFERENCE.md Pattern 4

### By Provider

**OpenAI**
- OpenAIParser
- normalize_json_response()
- Message format handling
- See FEATURES_ANALYSIS.md section 5

**Anthropic**
- AnthropicParser
- Content array handling
- Token mapping (input→prompt, output→completion)
- See FEATURES_ANALYSIS.md section 5

**Ollama**
- OllamaParser
- OllamaResponseHandler
- Streaming support
- Context window tracking
- See FEATURES_ANALYSIS.md sections 3 & 5

**Generic/Unknown**
- GenericParser
- Fallback support
- Auto-detection
- See FEATURES_ANALYSIS.md section 5

---

## Key Concepts

### Normalization (Input)
Processing a prompt before sending to LLM:
1. Format standardization
2. Type detection
3. Segmentation
4. Deduplication
5. Metadata enrichment

**Main class**: PromptProcessor  
**See**: FEATURES_ANALYSIS.md section 1

### Normalization (Output)
Processing an LLM response:
1. Provider format detection
2. Content extraction
3. Artifact removal
4. Code block standardization
5. Metadata extraction

**Main class**: ResponseNormalizer  
**See**: FEATURES_ANALYSIS.md section 3

### Bidirectional Context
Unified interface for input and output normalization:
```python
normalizer = ContextNormalizer()
normalizer.normalize_input(prompt)      # Input pipeline
normalizer.normalize_output(response)   # Output pipeline
```

**Main class**: ContextNormalizer  
**See**: FEATURES_ANALYSIS.md section 2

### Structured Representation
Converting between text and JSON while preserving information:
- Text → JSON (parse) with full structure
- JSON → Text (reconstruct) with fidelity control
- Lossless round-trip conversion

**Main class**: StructuredParser  
**See**: FEATURES_ANALYSIS.md section 4

### Fidelity Levels
Controlling information retention:
- MAXIMUM: Everything, all structure
- HIGH: Semantic structure, normalize formatting
- MEDIUM: Main content, discard auxiliary
- LOW: Summary only, discard details
- MINIMAL: Core message only

**See**: FEATURES_ANALYSIS.md section 4

### Provider Abstraction
Working with multiple LLMs transparently:
- Auto-detection of provider format
- Unified output format
- Provider-specific metadata preservation
- Per-provider configuration

**Main class**: UnifiedResponseParser  
**See**: FEATURES_ANALYSIS.md section 5

---

## Recommended Reading Order

### For Quick Integration (30 minutes)
1. QUICK_REFERENCE.md - "Core API Reference" sections 1-2
2. QUICK_REFERENCE.md - "Common Patterns" section 1
3. Copy code, integrate, done

### For Comprehensive Understanding (2-3 hours)
1. README.md - Overview
2. ARCHITECTURE_DIAGRAM.md - System overview + pipelines
3. FEATURES_ANALYSIS.md - Sections 1-3 (input/output/bidirectional)
4. QUICK_REFERENCE.md - Code examples
5. FEATURES_ANALYSIS.md - Sections 4-6 (remaining features)

### For Deep Architecture Review (4-5 hours)
1. ARCHITECTURE_DIAGRAM.md - All sections
2. FEATURES_ANALYSIS.md - All sections in detail
3. Review code comments in actual files
4. Run integration tests
5. Build proof-of-concept

### For Troubleshooting (5-15 minutes)
1. QUICK_REFERENCE.md - "Troubleshooting" table
2. Related section in FEATURES_ANALYSIS.md
3. Check configuration options
4. Review ARCHITECTURE_DIAGRAM.md for integration context

---

## File Statistics

| File | Lines | Sections | Content Type |
|------|-------|----------|--------------|
| QUICK_REFERENCE.md | 592 | 11 | Code + explanations |
| FEATURES_ANALYSIS.md | 1,267 | 6 major | Deep analysis |
| ARCHITECTURE_DIAGRAM.md | 435 | 7 | Diagrams + code |
| **Total** | **2,294** | **24** | Comprehensive |

---

## How to Stay Updated

Documentation is version-aligned with code:
- See CHANGELOG.md for recent changes
- Each major section has examples with recent patterns
- Performance characteristics are empirically validated
- Provider support reflects actual API versions

**Last Updated**: 2024-11-11  
**Documentation Version**: 1.0  
**Framework Versions**: All recent versions (see requirements.txt)

---

## Additional Resources

### In Repository
- **README.md**: Getting started, installation, quick examples
- **examples/**: Working code examples
- **tests/**: Test cases demonstrating usage
- **CONTRIBUTING.md**: Development guidelines

### In Code
- Module docstrings
- Function docstrings with examples
- Type hints throughout
- Inline comments for complex logic

### External
- Provider API documentation
  - [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
  - [Anthropic API Reference](https://docs.anthropic.com/en/api)
  - [Ollama Documentation](https://github.com/ollama/ollama)

---

## Quick Links Within Documentation

### FEATURES_ANALYSIS.md
- [PromptProcessor](#1-prompt-processing-features)
- [ContextNormalizer](#2-bidirectional-normalization)
- [ResponseNormalizer](#3-response-processing)
- [StructuredParser](#4-structured-parsing)
- [UnifiedResponseParser](#5-provider-parsers)
- [ContextSegmenter](#61-context-segmentation)
- [ContentClassifier](#62-content-classification)
- [ContentDeduplicator](#63-content-deduplication)

### ARCHITECTURE_DIAGRAM.md
- [System Architecture](#system-architecture-overview)
- [Input Pipeline](#input-processing-pipeline)
- [Output Pipeline](#output-processing-pipeline)
- [Class Hierarchy](#class-hierarchy--dependencies)
- [Performance](#performance-characteristics)
- [Integrations](#integration-points)

### QUICK_REFERENCE.md
- [ContextNormalizer](#1-bidirectional-normalization-recommended-entry-point)
- [PromptProcessor](#2-prompt-processing)
- [ResponseNormalizer](#3-response-processing)
- [Provider Parsers](#4-provider-parsers-auto-detection)
- [StructuredParser](#5-structured-parsing)
- [Segmenter](#6-text-segmentation)
- [Classifier](#7-content-classification)
- [Deduplicator](#8-content-deduplication)
- [Common Patterns](#common-patterns)
- [Configuration](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

**Questions?** Check the relevant section above, then search the documentation.
