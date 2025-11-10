"""
HydraContext - Intelligent context chunking for LLM memory systems.

A robust text segmentation, deduplication, and context normalization library
designed for Hydra's memory layer and LLM pipelines.
"""

__version__ = "0.1.0"
__author__ = "HydraContext Contributors"

# Core segmentation and deduplication
from hydracontext.core.segmenter import ContextSegmenter
from hydracontext.core.deduplicator import ContentDeduplicator
from hydracontext.core.classifier import ContentClassifier

# Prompt processing and normalization
from hydracontext.core.prompt_processor import normalize_prompt, split_prompt
from hydracontext.core.bidirectional import ContextNormalizer
from hydracontext.core.response_processor import ResponseNormalizer
from hydracontext.core.structured_parser import StructuredParser
from hydracontext.core.provider_parsers import UnifiedResponseParser

# API access
from hydracontext.api import HydraContextAPI

__all__ = [
    # Core functionality
    "ContextSegmenter",
    "ContentDeduplicator",
    "ContentClassifier",
    # Prompt processing
    "normalize_prompt",
    "split_prompt",
    "ContextNormalizer",
    "ResponseNormalizer",
    "StructuredParser",
    "UnifiedResponseParser",
    # API
    "HydraContextAPI",
]
