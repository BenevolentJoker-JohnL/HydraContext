"""
Custom exceptions for HydraContext.

Provides structured error handling with specific exception types.
"""


class HydraContextError(Exception):
    """Base exception for all HydraContext errors."""
    pass


class ValidationError(HydraContextError):
    """Raised when input validation fails."""
    pass


class ParsingError(HydraContextError):
    """Raised when parsing fails."""
    pass


class ProviderError(HydraContextError):
    """Raised when provider-specific operations fail."""
    pass


class NormalizationError(HydraContextError):
    """Raised when normalization fails."""
    pass


class SegmentationError(HydraContextError):
    """Raised when text segmentation fails."""
    pass


class FidelityError(HydraContextError):
    """Raised when fidelity operations fail."""
    pass


class ConfigurationError(HydraContextError):
    """Raised when configuration is invalid."""
    pass
