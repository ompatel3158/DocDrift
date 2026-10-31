"""Base verifier class for DocDrift."""

from abc import ABC, abstractmethod
from docdrift.models import Claim, VerificationResult


class BaseVerifier(ABC):
    """Abstract base class for all DocDrift verifiers."""

    @abstractmethod
    def can_verify(self, claim: Claim) -> bool:
        """Determines whether this verifier can handle the given claim."""
        pass

    @abstractmethod
    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies the claim against the repository at repo_path.

        Args:
            claim: The claim extracted from documentation.
            repo_path: Absolute or relative path to the repository root.

        Returns:
            VerificationResult with verdict, evidence, and optional suggested_value.
        """
        pass
