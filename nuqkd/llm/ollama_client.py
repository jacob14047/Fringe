"""
nuqkd.llm.ollama_client
=========================
HTTP client for a locally-running Ollama inference server.

Ollama exposes a REST API at http://localhost:11434 by default.
Models are pulled once and cached locally — the first call to a new model
triggers an automatic pull if the model is available.

Usage::

    client = OllamaClient(model="gemma3:12b")
    response = client.chat([
        {"role": "system", "content": "You are a quantum security expert."},
        {"role": "user",   "content": "What is a PNS attack?"}
    ])

Structured output::

    schema = {"type": "object", "properties": {"attack": {"type": "string"}}}
    data   = client.chat_json(messages, schema)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    Parameters
    ----------
    model : str
        Model name as known to Ollama, e.g. ``"gemma3:12b"``,
        ``"qwen2.5:14b"``, ``"mistral:7b-instruct"``.
    base_url : str
        Ollama server address.
    temperature : float
        Sampling temperature (0 = greedy, 1 = creative).
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Number of retries on transient errors.
    """

    SUPPORTED_MODELS = [
        "gemma3:12b", "gemma3:27b",
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
        "mistral:7b-instruct", "mistral-nemo:12b",
        "llama3.2:3b", "llama3.1:8b", "llama3.1:70b",
        "phi4:14b", "deepseek-r1:14b",
    ]

    def __init__(self,
                 model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.2,
                 timeout: int = 300,
                 max_retries: int = 3,
                 system_prompt: str = "") -> None:
        self.model         = model
        self.base_url      = base_url.rstrip("/")
        self.temperature   = temperature
        self.timeout       = timeout
        self.max_retries   = max_retries
        self.system_prompt = system_prompt
        self._session      = requests.Session()

    # ------------------------------------------------------------------
    # Core chat completion
    # ------------------------------------------------------------------

    def chat(self,
             messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: int = 2048) -> str:
        """
        Send a chat completion request and return the assistant's response text.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts
        temperature : optional override
        max_tokens : maximum tokens to generate

        Returns
        -------
        str — the assistant's response text
        """
        all_messages = []
        if self.system_prompt:
            all_messages.append({"role": "system", "content": self.system_prompt})
        all_messages.extend(messages)

        payload = {
            "model":   self.model,
            "messages": all_messages,
            "stream":  False,
            "options": {
                "temperature":  temperature if temperature is not None else self.temperature,
                "num_predict":  max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama not reachable at %s — is it running?", self.base_url)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except requests.exceptions.Timeout:
                logger.warning("Ollama timeout (attempt %d/%d)", attempt + 1, self.max_retries)
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                else:
                    raise
            except Exception as exc:
                logger.error("Ollama error: %s", exc)
                raise

        return ""   # unreachable

    def chat_json(self,
                  messages: List[Dict[str, str]],
                  schema_hint: Optional[Dict] = None,
                  temperature: float = 0.1) -> Dict[str, Any]:
        """
        Request a JSON-structured response.

        Appends a strict instruction to the last user message asking for
        pure JSON output conforming to the optional schema hint.
        Returns parsed dict, or ``{"error": ..., "raw": ...}`` on failure.
        """
        # Inject JSON instruction
        messages = list(messages)
        schema_str = json.dumps(schema_hint, indent=2) if schema_hint else "a JSON object"
        json_instruction = (
            f"\n\nRESPOND WITH ONLY VALID JSON MATCHING THIS SCHEMA:\n{schema_str}"
            "\nDo not include any explanation, markdown, or code fences. "
            "Return only the raw JSON object."
        )
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"] + json_instruction,
            }

        raw = self.chat(messages, temperature=temperature)

        # Strip markdown fences if model adds them anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s\nRaw: %s", exc, raw[:200])
            return {"error": str(exc), "raw": raw}

    # ------------------------------------------------------------------
    # Convenience: single-turn prompt
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return names of locally-available models."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def pull_model(self, model: Optional[str] = None) -> bool:
        """Pull a model if not already cached."""
        target = model or self.model
        try:
            resp = self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": target, "stream": False},
                timeout=600,
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Pull failed: %s", exc)
            return False

    def is_available(self) -> bool:
        """Check whether Ollama server is running."""
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model!r}, url={self.base_url!r})"
