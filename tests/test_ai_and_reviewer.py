"""Unit tests for Gemma 4 AI Client, Claim Extractor, Fixer, and Reviewer."""

import os
from unittest.mock import MagicMock, patch
import pytest

from docdrift.ai_client import DEFAULT_MODEL, SUPPORTED_MODELS, GemmaClient, validate_gemma_model
from docdrift.extractor import (
    ExtractedClaimItem,
    ExtractionResponse,
    GemmaClaimExtractor,
    HybridExtractor,
    RuleBasedExtractor,
)
from docdrift.fixer import FixResult, GemmaFixer, HybridFixer, RuleBasedFixer
from docdrift.models import Claim, ClaimType, DocSection, Verdict
from docdrift.reviewer import InteractiveReviewer, apply_fix_to_file
from google.genai import errors


# ============================================================================
# Gemma 4 AI Client Tests
# ============================================================================


def test_ai_client_supported_models():
    """Verify DEFAULT_MODEL is gemma-4-26b-it and all SUPPORTED_MODELS are valid."""
    assert DEFAULT_MODEL == "gemma-4-26b-it"
    assert "gemma-4-26b-it" in SUPPORTED_MODELS
    assert "gemma-4-31b-it" in SUPPORTED_MODELS
    assert "gemma-4-26b" in SUPPORTED_MODELS
    assert "gemma-4-31b" in SUPPORTED_MODELS

    for model_name in SUPPORTED_MODELS:
        validate_gemma_model(model_name)
        client = GemmaClient(model=model_name, offline=True)
        assert client.model == model_name


def test_ai_client_strictly_rejects_non_gemma_models():
    """Verify that non-Gemma models are strictly rejected with ValueError."""
    unsupported = [
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gpt-4o",
        "claude-3-5-sonnet",
        "gemma-2-9b",
        "gemma-2-27b",
        "llama-3-70b",
    ]
    for model in unsupported:
        with pytest.raises(ValueError) as excinfo:
            GemmaClient(model=model, offline=True)
        assert "not a supported Gemma 4 model" in str(excinfo.value)


def test_ai_client_offline_mode_detection(monkeypatch):
    """Verify offline mode triggers when explicitly requested or GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # No env var, no explicit api_key -> offline
    client1 = GemmaClient()
    assert client1.is_offline is True

    # Explicit offline=True
    client2 = GemmaClient(api_key="dummy_key", offline=True)
    assert client2.is_offline is True

    # With API key and offline=False
    client3 = GemmaClient(api_key="test-key", offline=False)
    assert client3.offline is False
    assert client3.is_offline is False


def test_ai_client_offline_text_generation():
    """Test text generation with mock provider and mock response in offline mode."""
    client = GemmaClient(offline=True)

    # Static mock response
    client.set_mock_response("Fixed line replacement")
    res = client.generate_text("Prompt")
    assert res == "Fixed line replacement"

    # Dynamic mock provider
    client.set_mock_provider(lambda prompt, schema: f"Echo: {prompt}")
    res2 = client.generate_text("Hello Gemma")
    assert res2 == "Echo: Hello Gemma"


def test_ai_client_offline_structured_json():
    """Test structured JSON generation adhering to Pydantic schema in offline mode."""
    client = GemmaClient(offline=True)

    class CustomSchema(FixResult):
        pass

    # With static dict mock response
    client.set_mock_response({"fixed_line": "  const PORT = 3000;", "explanation": "Updated port"})
    result = client.generate_structured_json("Prompt", CustomSchema)
    assert isinstance(result, CustomSchema)
    assert result.fixed_line == "  const PORT = 3000;"
    assert result.explanation == "Updated port"

    # Fallback when no mock is set
    client_empty = GemmaClient(offline=True)
    res_empty = client_empty.generate_structured_json("Prompt", CustomSchema)
    assert isinstance(res_empty, CustomSchema)


def test_ai_client_retry_logic():
    """Test that transient API errors trigger retries."""
    client = GemmaClient(api_key="test-key", offline=False, max_retries=2, retry_delay=0.01)

    mock_client = MagicMock()
    # First call fails with ServerError (500), second succeeds
    mock_resp = MagicMock()
    mock_resp.text = "Success on retry"
    mock_client.models.generate_content.side_effect = [
        errors.ServerError(500, {"message": "Temporary internal server error"}),
        mock_resp,
    ]
    client._client = mock_client

    result = client.generate_text("Test prompt")
    assert result == "Success on retry"
    assert mock_client.models.generate_content.call_count == 2


def test_ai_client_unrecoverable_failure():
    """Test that exhausting retries raises RuntimeError."""
    client = GemmaClient(api_key="test-key", offline=False, max_retries=2, retry_delay=0.01)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ServerError(500, {"message": "Persistent failure"})
    client._client = mock_client

    with pytest.raises(RuntimeError) as excinfo:
        client.generate_text("Test prompt")
    assert "failed after 2 retries" in str(excinfo.value)


# ============================================================================
# Claim Extractor Tests
# ============================================================================


def test_rule_based_extractor_claims():
    """Verify deterministic extraction of all 4 claim types."""
    section = DocSection(
        doc_path="README.md",
        section_title="Getting Started",
        start_line=1,
        lines=[
            "# Setup Guide",
            "1. First inspect the config in `config/default.yaml` and `src/server.ts`.",
            "2. Also ensure `Dockerfile` is present in root.",
            "3. Run `npm run test` or `cargo build --release` to verify compilation.",
            "4. Configure environment with `DATABASE_URL` and `PORT`.",
            "5. The server listens on `8080:8080` or docker run -p 3000:3000.",
        ],
    )

    extractor = RuleBasedExtractor()
    claims = extractor.extract(section)

    claim_types = [c.claim_type for c in claims]
    assert ClaimType.FILE_PATH in claim_types
    assert ClaimType.COMMAND_SCRIPT in claim_types
    assert ClaimType.ENV_VAR in claim_types
    assert ClaimType.DOCKER_PORT in claim_types

    extracted_values = [c.extracted_value for c in claims]
    assert "config/default.yaml" in extracted_values
    assert "src/server.ts" in extracted_values
    assert "Dockerfile" in extracted_values
    assert "npm run test" in extracted_values
    assert "cargo build --release" in extracted_values
    assert "DATABASE_URL" in extracted_values
    assert "PORT" in extracted_values
    assert "8080:8080" in extracted_values
    assert "3000:3000" in extracted_values

    # Check line numbers match section 1-based indexing
    file_claim = next(c for c in claims if c.extracted_value == "config/default.yaml")
    assert file_claim.line_number == 2
    assert file_claim.doc_path == "README.md"
    assert file_claim.section_title == "Getting Started"


def test_rule_based_extractor_line_level_patterns():
    """Verify extraction of unbackticked CLI prompts and EXPOSE/export statements."""
    section = DocSection(
        doc_path="DOCKER.md",
        section_title="Docker Setup",
        start_line=10,
        lines=[
            "$ docker build -t myapp .",
            "EXPOSE 8000",
            "export API_SECRET=supersecret",
            "> cargo test --workspace",
        ],
    )
    extractor = RuleBasedExtractor()
    claims = extractor.extract(section)

    vals = {c.extracted_value: c.claim_type for c in claims}
    assert "docker build -t myapp ." in vals
    assert vals["docker build -t myapp ."] == ClaimType.COMMAND_SCRIPT

    assert "8000" in vals
    assert vals["8000"] == ClaimType.DOCKER_PORT

    assert "API_SECRET" in vals
    assert vals["API_SECRET"] == ClaimType.ENV_VAR

    assert "cargo test --workspace" in vals
    assert vals["cargo test --workspace"] == ClaimType.COMMAND_SCRIPT


def test_gemma_claim_extractor():
    """Verify GemmaClaimExtractor parses structured output into Claim models."""
    client = GemmaClient(offline=True)
    mock_payload = ExtractionResponse(
        claims=[
            ExtractedClaimItem(
                line_number=5,
                raw_line="Run `npm test`",
                claim_type="command_script",
                extracted_value="npm test",
            ),
            ExtractedClaimItem(
                line_number=7,
                raw_line="Connect to `DATABASE_URL`",
                claim_type="env_var",
                extracted_value="DATABASE_URL",
            ),
        ]
    )
    client.set_mock_response(mock_payload)

    section = DocSection(
        doc_path="docs/setup.md",
        section_title="Installation",
        start_line=1,
        lines=["Line 1", "Line 2", "Line 3", "Line 4", "Run `npm test`", "Line 6", "Connect to `DATABASE_URL`"],
    )

    extractor = GemmaClaimExtractor(client)
    claims = extractor.extract(section)

    assert len(claims) == 2
    assert claims[0].claim_type == ClaimType.COMMAND_SCRIPT
    assert claims[0].extracted_value == "npm test"
    assert claims[0].line_number == 5

    assert claims[1].claim_type == ClaimType.ENV_VAR
    assert claims[1].extracted_value == "DATABASE_URL"
    assert claims[1].line_number == 7


def test_hybrid_extractor_fallback():
    """Test that HybridExtractor falls back to RuleBasedExtractor when client is offline."""
    client = GemmaClient(offline=True)
    extractor = HybridExtractor(client=client)

    section = DocSection(
        doc_path="README.md",
        section_title="Commands",
        start_line=10,
        lines=["Execute `cargo test` to run unit tests."],
    )

    claims = extractor.extract(section)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.COMMAND_SCRIPT
    assert claims[0].extracted_value == "cargo test"
    assert claims[0].line_number == 10


# ============================================================================
# Fixer Tests
# ============================================================================


def test_rule_based_fixer_replacement():
    """Verify RuleBasedFixer performs clean surgical replacement."""
    fixer = RuleBasedFixer()

    claim = Claim(
        doc_path="README.md",
        line_number=4,
        raw_line="  Run `npm run test:legacy` to execute tests.",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run test:legacy",
        verdict=Verdict.DRIFTED,
        suggested_replacement="npm test",
    )

    fixed = fixer.fix_claim(claim)
    assert fixed == "  Run `npm test` to execute tests."
    assert claim.fixed_line == "  Run `npm test` to execute tests."


def test_rule_based_fixer_evidence_inference():
    """Verify RuleBasedFixer infers suggested replacement from evidence."""
    fixer = RuleBasedFixer()

    claim = Claim(
        doc_path="README.md",
        line_number=8,
        raw_line="See `src/utils.old.ts` for helpers.",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="src/utils.old.ts",
        verdict=Verdict.DRIFTED,
        evidence="File 'src/utils.old.ts' not found; actual file: `src/utils.ts`",
    )

    fixed = fixer.fix_claim(claim)
    assert fixed == "See `src/utils.ts` for helpers."


def test_gemma_fixer_and_hybrid():
    """Verify GemmaFixer uses AI client and preserves leading whitespace."""
    client = GemmaClient(offline=True)
    client.set_mock_response(FixResult(fixed_line="const PORT = 4000;", explanation="Updated port"))

    claim = Claim(
        doc_path="server.md",
        line_number=12,
        raw_line="    const PORT = 3000;",
        claim_type=ClaimType.DOCKER_PORT,
        extracted_value="3000",
        verdict=Verdict.DRIFTED,
        suggested_replacement="4000",
    )

    fixer = GemmaFixer(client)
    fixed = fixer.fix_claim(claim)

    # Check leading indentation was preserved
    assert fixed == "    const PORT = 4000;"
    assert claim.fixed_line == "    const PORT = 4000;"

    # Test HybridFixer with offline client defaults to rule fixer
    hybrid = HybridFixer(client=GemmaClient(offline=True))
    fixed_hybrid = hybrid.fix_claim(claim)
    assert "4000" in fixed_hybrid


# ============================================================================
# Reviewer & In-Place File Update Tests
# ============================================================================


def test_apply_fix_to_file(tmp_path):
    """Test in-place file replacement preserving indentation and newline style."""
    test_file = tmp_path / "TEST_DOC.md"

    # Unix newlines
    content = "Line 1\n  Line 2 with `old_token`\nLine 3\n"
    test_file.write_text(content, encoding="utf-8", newline="")

    success = apply_fix_to_file(str(test_file), 2, "  Line 2 with `new_token`")
    assert success is True

    updated = test_file.read_text(encoding="utf-8")
    assert updated == "Line 1\n  Line 2 with `new_token`\nLine 3\n"

    # Windows CRLF newlines
    crlf_content = "First\r\n    Run `npm test:old`\r\nThird\r\n"
    test_file.write_bytes(crlf_content.encode("utf-8"))

    success2 = apply_fix_to_file(str(test_file), 2, "Run `npm test`")
    assert success2 is True

    updated_bytes = test_file.read_bytes()
    # Check exact indentation and CRLF preserved
    assert updated_bytes == b"First\r\n    Run `npm test`\r\nThird\r\n"


def test_apply_fix_to_file_edge_cases(tmp_path):
    """Test out-of-bounds line numbers and missing file paths return False."""
    assert apply_fix_to_file("non_existent_file.md", 1, "test") is False

    test_file = tmp_path / "small.md"
    test_file.write_text("Line 1\nLine 2\n", encoding="utf-8")
    assert apply_fix_to_file(str(test_file), 999, "out of bounds") is False
    assert apply_fix_to_file(str(test_file), 0, "out of bounds") is False


def test_reviewer_split_diff_view():
    """Verify Split-Diff visual component rendering."""
    claim = Claim(
        doc_path="README.md",
        line_number=15,
        raw_line="Run `npm test:unit` for all tests.",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm test:unit",
        verdict=Verdict.DRIFTED,
        evidence="Script 'test:unit' does not exist in package.json",
        suggested_replacement="npm test",
        fixed_line="Run `npm test` for all tests.",
    )

    reviewer = InteractiveReviewer(non_interactive=True)
    table = reviewer.create_split_diff_view(claim, index=1, total=1)
    assert table is not None


def test_reviewer_empty_claims():
    """Verify review_claims handles empty or non-drifted lists cleanly."""
    reviewer = InteractiveReviewer(non_interactive=True)
    stats = reviewer.review_claims([])
    assert stats["total"] == 0
    assert stats["accepted"] == 0

    verified_claim = Claim(
        doc_path="README.md",
        line_number=1,
        raw_line="Check `src/index.ts`",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="src/index.ts",
        verdict=Verdict.VERIFIED,
    )
    stats2 = reviewer.review_claims([verified_claim])
    assert stats2["total"] == 0


def test_reviewer_non_interactive_accept(tmp_path):
    """Test non-interactive mode automatically accepting drifted claim fixes."""
    doc = tmp_path / "SETUP.md"
    doc.write_text("Header\nCheck `src/v1.py`\nFooter\n", encoding="utf-8")

    claim = Claim(
        doc_path=str(doc),
        line_number=2,
        raw_line="Check `src/v1.py`",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="src/v1.py",
        verdict=Verdict.DRIFTED,
        evidence="Path v1.py was renamed to v2.py",
        suggested_replacement="src/v2.py",
        fixed_line="Check `src/v2.py`",
    )

    reviewer = InteractiveReviewer(non_interactive=True, auto_action="accept")
    stats = reviewer.review_claims([claim])

    assert stats["total"] == 1
    assert stats["accepted"] == 1
    assert stats["skipped"] == 0

    updated_text = doc.read_text(encoding="utf-8")
    assert "Check `src/v2.py`" in updated_text


def test_reviewer_non_interactive_skip(tmp_path):
    """Test non-interactive mode skipping fixes."""
    doc = tmp_path / "SETUP.md"
    orig_content = "Header\nCheck `src/v1.py`\nFooter\n"
    doc.write_text(orig_content, encoding="utf-8")

    claim = Claim(
        doc_path=str(doc),
        line_number=2,
        raw_line="Check `src/v1.py`",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="src/v1.py",
        verdict=Verdict.DRIFTED,
        suggested_replacement="src/v2.py",
    )

    reviewer = InteractiveReviewer(non_interactive=True, auto_action="skip")
    stats = reviewer.review_claims([claim])

    assert stats["total"] == 1
    assert stats["accepted"] == 0
    assert stats["skipped"] == 1

    # File should remain unchanged
    assert doc.read_text(encoding="utf-8") == orig_content


def test_reviewer_interactive_custom_inputs(tmp_path):
    """Test interactive review actions (accept, edit, skip, quit) with simulated inputs."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "Line 1\nLine 2: `old1`\nLine 3: `old2`\nLine 4: `old3`\nLine 5: `old4`\n",
        encoding="utf-8",
    )

    claims = [
        Claim(
            doc_path=str(doc),
            line_number=2,
            raw_line="Line 2: `old1`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="old1",
            verdict=Verdict.DRIFTED,
            suggested_replacement="new1",
            fixed_line="Line 2: `new1`",
        ),
        Claim(
            doc_path=str(doc),
            line_number=3,
            raw_line="Line 3: `old2`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="old2",
            verdict=Verdict.DRIFTED,
            suggested_replacement="new2",
            fixed_line="Line 3: `new2`",
        ),
        Claim(
            doc_path=str(doc),
            line_number=4,
            raw_line="Line 4: `old3`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="old3",
            verdict=Verdict.DRIFTED,
            suggested_replacement="new3",
            fixed_line="Line 4: `new3`",
        ),
        Claim(
            doc_path=str(doc),
            line_number=5,
            raw_line="Line 5: `old4`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="old4",
            verdict=Verdict.DRIFTED,
            suggested_replacement="new4",
            fixed_line="Line 5: `new4`",
        ),
    ]

    # Sequence of user responses:
    # Invalid input first -> 'invalid', then 'a' (accept)
    # Claim 2 -> 'e' (edit) -> 'Line 3: `manually_edited`'
    # Claim 3 -> 's' (skip)
    # Claim 4 -> 'q' (quit)
    inputs = ["invalid", "a", "e", "Line 3: `manually_edited`", "s", "q"]

    def mock_input(prompt: str) -> str:
        return inputs.pop(0)

    reviewer = InteractiveReviewer(non_interactive=False, input_func=mock_input)
    stats = reviewer.review_claims(claims)

    assert stats["accepted"] == 1
    assert stats["edited"] == 1
    assert stats["skipped"] == 1

    content = doc.read_text(encoding="utf-8")
    assert "Line 2: `new1`" in content
    assert "Line 3: `manually_edited`" in content
    assert "Line 4: `old3`" in content  # skipped
    assert "Line 5: `old4`" in content  # quit before reaching
