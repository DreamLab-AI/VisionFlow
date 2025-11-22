"""
Ontology Tools Shared Library
==============================

Common utilities and parsers for ontology conversion tools.
"""

from .ontology_block_parser import (
    OntologyBlock,
    OntologyBlockParser,
    DOMAIN_CONFIG,
    STANDARD_NAMESPACES
)

from .ontology_loader import (
    OntologyLoader,
    LoaderStatistics
)

__all__ = [
    'OntologyBlock',
    'OntologyBlockParser',
    'DOMAIN_CONFIG',
    'STANDARD_NAMESPACES',
    'OntologyLoader',
    'LoaderStatistics'
]
