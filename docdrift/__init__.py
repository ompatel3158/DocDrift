"""DocDrift: AI + deterministic code verifier for documentation."""

from docdrift.ai_client import DEFAULT_MODEL, SUPPORTED_MODELS, GemmaClient
from docdrift.extractor import (
    GemmaClaimExtractor,
    HybridExtractor,
    RuleBasedExtractor,
)
from docdrift.fixer import GemmaFixer, HybridFixer, RuleBasedFixer
from docdrift.models import Claim, ClaimType, DocSection, DriftReport, Verdict, VerificationResult
from docdrift.reviewer import InteractiveReviewer, apply_fix_to_file

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "SUPPORTED_MODELS",
    "GemmaClient",
    "GemmaClaimExtractor",
    "RuleBasedExtractor",
    "HybridExtractor",
    "GemmaFixer",
    "RuleBasedFixer",
    "HybridFixer",
    "InteractiveReviewer",
    "apply_fix_to_file",
    "Claim",
    "ClaimType",
    "DocSection",
    "DriftReport",
    "Verdict",
    "VerificationResult",
]
