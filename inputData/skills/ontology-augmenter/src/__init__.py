"""
Ontology Augmenter - Perplexity API Integration
Enriches ontology content with current research and citations
"""

from .perplexity_enricher import (
    PerplexityEnricher,
    EnrichmentResult
)

__all__ = [
    "PerplexityEnricher",
    "EnrichmentResult"
]

__version__ = "1.0.0"
