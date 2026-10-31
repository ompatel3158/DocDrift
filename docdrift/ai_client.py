"""Gemma 4 AI Client for DocDrift.

Strictly configured for Gemma 4 models using the official google-genai SDK.
Supports structured JSON extraction, surgical documentation line rewrites,
retry with exponential backoff, and robust offline/mock mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, List, Optional, Set, Type, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

logger = logging.getLogger("docdrift.ai_client")

# Strict Gemma 4 Model Configuration
def load_env_file(search_paths: Optional[List[str]] = None) -> None:
    """Zero-dependency .env file loader for DocDrift."""
    paths_to_check = search_paths or [
        os.path.join(os.getcwd(), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
    ]
    for p in paths_to_check:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


load_env_file()

DEFAULT_MODEL: str = os.environ.get("DOCDRIFT_MODEL", "gemma-4-31b-it")
SUPPORTED_MODELS: Set[str] = {
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-26b",
    "gemma-4-31b",
}

def resolve_model_name(model: str) -> str:
    """Normalize model identifier for Gemini API."""
    if model in ("gemma-4-26b-it", "gemma-4-26b"):
        return "gemma-4-26b-a4b-it"
    return model

T = TypeVar("T")


def validate_gemma_model(model: str) -> None:
    """Validate that the given model is an approved Gemma 4 model.

    Raises:
        ValueError: If model is not in SUPPORTED_MODELS.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model}' is not a supported Gemma 4 model. "
            f"DocDrift is strictly configured for Gemma 4 models: {sorted(SUPPORTED_MODELS)}"
        )


def _strip_markdown_code_fence(text: str) -> str:
    """Strip markdown code fence wrapper (```json ... ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


class GemmaClient:
    """Official Google GenAI client configured strictly for Gemma 4."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        offline: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize GemmaClient.

        Args:
            model: The Gemma 4 model identifier. Must be in SUPPORTED_MODELS.
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            offline: If True, forces offline mode regardless of API key presence.
            max_retries: Number of retries for transient API errors.
            retry_delay: Initial retry delay in seconds for exponential backoff.
        """
        validate_gemma_model(model)
        self.model = resolve_model_name(model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.offline = offline or not bool(self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Mock handlers for offline testing and simulations
        self._mock_provider: Optional[Callable[[str, Optional[Any]], Any]] = None
        self._mock_response: Optional[Any] = None

        self._client: Optional[genai.Client] = None
        if not self.offline:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize google-genai Client ({e}). Falling back to offline mode."
                )
                self.offline = True
                self._client = None

    @property
    def is_offline(self) -> bool:
        """Return True if client is operating in offline/mock mode."""
        return self.offline

    def set_mock_provider(self, provider: Optional[Callable[[str, Optional[Any]], Any]]) -> None:
        """Set a dynamic mock provider callback (prompt, response_schema) -> response."""
        self._mock_provider = provider

    def set_mock_response(self, response: Any) -> None:
        """Set a static mock response for offline queries."""
        self._mock_response = response

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Gemma 4 with retries and offline support.

        Args:
            prompt: User prompt content.
            system_instruction: Optional system instruction / system prompt.
            temperature: Sampling temperature (0.0 to 2.0).
            max_output_tokens: Optional maximum token count.

        Returns:
            The generated string response.
        """
        if self.is_offline or self._client is None:
            return self._handle_offline_text(prompt)

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_kwargs)

        last_err: Optional[Exception] = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    return response.text
                return ""
            except (errors.APIError, errors.ServerError, ConnectionError, TimeoutError) as e:
                last_err = e
                logger.warning(
                    f"Gemma 4 API call attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
            except Exception as e:
                # Non-retryable client error or configuration problem
                logger.error(f"Unrecoverable Gemma 4 error: {e}")
                raise

        raise RuntimeError(
            f"Gemma 4 text generation failed after {self.max_retries} retries: {last_err}"
        ) from last_err

    def generate_structured_json(
        self,
        prompt: str,
        response_schema: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Any:
        """Generate structured JSON output adhering to response_schema.

        Args:
            prompt: User prompt text.
            response_schema: A Pydantic BaseModel class, Python type, or JSON schema.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (defaults to 0.1 for high determinism).

        Returns:
            Validated Pydantic model instance or parsed JSON data (dict/list).
        """
        if self.is_offline or self._client is None:
            return self._handle_offline_structured(prompt, response_schema)

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        last_err: Optional[Exception] = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                raw_text = response.text or "{}"
                cleaned = _strip_markdown_code_fence(raw_text)
                parsed_json = json.loads(cleaned)

                # If response_schema is a Pydantic model, validate and return instance
                if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                    return response_schema.model_validate(parsed_json)

                return parsed_json
            except (errors.APIError, errors.ServerError, ConnectionError, TimeoutError) as e:
                last_err = e
                logger.warning(
                    f"Gemma 4 structured call attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning(
                    f"Failed to parse JSON response on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
            except Exception as e:
                logger.error(f"Unrecoverable structured JSON error: {e}")
                raise

        raise RuntimeError(
            f"Gemma 4 structured generation failed after {self.max_retries} retries: {last_err}"
        ) from last_err

    def _handle_offline_text(self, prompt: str) -> str:
        """Return mock response when client is offline."""
        if self._mock_provider:
            val = self._mock_provider(prompt, None)
            return str(val) if val is not None else ""
        if self._mock_response is not None:
            return str(self._mock_response)
        return ""

    def _handle_offline_structured(self, prompt: str, response_schema: Any) -> Any:
        """Return mock structured response when client is offline."""
        if self._mock_provider:
            return self._mock_provider(prompt, response_schema)
        if self._mock_response is not None:
            if isinstance(self._mock_response, (dict, list)):
                if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                    return response_schema.model_validate(self._mock_response)
                return self._mock_response
            if isinstance(self._mock_response, BaseModel):
                return self._mock_response
            if isinstance(self._mock_response, str):
                parsed = json.loads(_strip_markdown_code_fence(self._mock_response))
                if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                    return response_schema.model_validate(parsed)
                return parsed

        # Sensible fallback
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            try:
                fields = {k: "" for k in response_schema.model_fields}
                return response_schema.model_construct(**fields)
            except Exception:
                try:
                    return response_schema.model_construct()
                except Exception:
                    return {}
        if isinstance(response_schema, list) or getattr(response_schema, "__origin__", None) is list:
            return []
        return {}


# Alias for backward/naming compatibility
GemmaAIClient = GemmaClient

