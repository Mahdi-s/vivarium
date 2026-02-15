from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

# Mock torch.library.register_fake if missing (to fix torchvision/torch mismatch)
try:
    import torch
    if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
        def _register_fake(name):
            def decorator(fn): return fn
            return decorator
        torch.library.register_fake = _register_fake
except ImportError:
    pass

# Mock warn_once in torch._dynamo.utils if missing (to fix torchao/torch mismatch)
try:
    import torch._dynamo.utils as _du
    if not hasattr(_du, "warn_once"):
        def _warn_once(msg): pass
        _du.warn_once = _warn_once
except (ImportError, AttributeError):
    pass

from aam.tools import ToolSpec

# For HuggingFaceTransformersGateway
try:
    import torch  # noqa: F401
except ImportError:
    pass  # Will raise error when gateway is used


JsonDict = Dict[str, Any]

if TYPE_CHECKING:  # pragma: no cover
    from aam.interpretability import CaptureContext


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting and backpressure handling."""

    max_concurrent_requests: int = 10
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    max_retries: int = 3
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    enable_context_degradation: bool = True
    context_degradation_threshold: int = 8000  # tokens

    @classmethod
    def default(cls) -> "RateLimitConfig":
        """
        Create default rate limit configuration with reasonable limits.

        Defaults:
        - 10 concurrent requests
        - 60 requests per minute (1 req/sec)
        - 100,000 tokens per minute (reasonable for most APIs)
        - Context degradation enabled at 8k tokens
        """
        return cls(
            max_concurrent_requests=10,
            requests_per_minute=60,
            tokens_per_minute=100000,
            max_retries=3,
            initial_backoff_s=1.0,
            max_backoff_s=60.0,
            enable_context_degradation=True,
            context_degradation_threshold=8000,
        )


def select_local_gateway(
    *,
    model_id_or_path: str,
    capture_context: Optional["CaptureContext"] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 128,
    prefer_transformerlens: bool = True,
    scientific_mode: bool = False,
) -> Any:
    """
    Choose the best local gateway for a model.

    Rationale:
    - OLMo-3 is not reliably supported by TransformerLens; prefer HuggingFaceHookedGateway.
    - For other models, prefer TransformerLens when available for richer tooling.
    
    Args:
        model_id_or_path: HuggingFace model ID or local path (or GGUF path)
        capture_context: Optional CaptureContext for activation capture
        device: Target device (cuda, mps, cpu) 
        max_new_tokens: Maximum tokens to generate
        prefer_transformerlens: If True, try TransformerLens first
        scientific_mode: If True, enforce that activation capture uses the same model
                         as inference. GGUF models will be rejected if capture_context
                         is set, as they cannot be probed. This prevents the "dual-stack"
                         validity threat where simulation uses GGUF but analysis uses PyTorch.
                         
    Raises:
        ValueError: If scientific_mode=True and a GGUF model is specified with capture_context
    """
    mid = str(model_id_or_path)
    
    # Enforce scientific mode: reject GGUF models when activation capture is enabled
    if scientific_mode and capture_context is not None:
        if mid.lower().endswith(".gguf"):
            raise ValueError(
                "Scientific mode requires PyTorch model for activation capture. "
                "GGUF models cannot be probed - the quantized weights differ from "
                "the full-precision PyTorch weights, invalidating interpretability findings. "
                "Use a HuggingFace model ID instead (e.g., 'allenai/Olmo-3-7B-Instruct')."
            )
    
    if "olmo" in mid.lower():
        return HuggingFaceHookedGateway(
            model_id_or_path=mid,
            device=device,
            capture_context=capture_context,
            max_new_tokens=int(max_new_tokens),
        )

    if prefer_transformerlens:
        try:
            return TransformerLensGateway(
                model_id=mid,
                device=device,
                capture_context=capture_context,
                max_new_tokens=int(max_new_tokens),
            )
        except Exception:
            # Fall back to HF if TL isn't installed / model unsupported.
            return HuggingFaceHookedGateway(
                model_id_or_path=mid,
                device=device,
                capture_context=capture_context,
                max_new_tokens=int(max_new_tokens),
            )

    return HuggingFaceHookedGateway(
        model_id_or_path=mid,
        device=device,
        capture_context=capture_context,
        max_new_tokens=int(max_new_tokens),
    )


class RateLimiter:
    """
    Rate limiter for LLM gateway with token counting, backpressure, and exponential backoff.

    Implements PRD Section 8.1 requirements:
    - Token counting before calls
    - Semaphore for concurrency control
    - 429 error handling with exponential backoff
    - Context degradation mode
    """

    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._request_times: List[float] = []
        self._token_budget: Optional[int] = config.tokens_per_minute
        self._token_window_start: float = time.time()
        self._token_window_tokens: int = 0

    def _estimate_tokens(self, messages: List[JsonDict], tools: Optional[List[ToolSpec]] = None) -> int:
        """
        Estimate token count for a request (rough approximation: 4 chars per token).
        """
        total_chars = 0
        for msg in messages:
            content = str(msg.get("content", ""))
            total_chars += len(content)
        if tools:
            for tool in tools:
                total_chars += len(tool.name) + len(tool.description)
                total_chars += len(json.dumps(tool.parameters))
        return max(1, total_chars // 4)

    async def _wait_for_rate_limit(self) -> None:
        """Wait if we've exceeded requests per minute."""
        if self._config.requests_per_minute is None:
            return

        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60.0]

        if len(self._request_times) >= self._config.requests_per_minute:
            # Wait until the oldest request is 60 seconds old
            oldest = min(self._request_times)
            wait_time = 60.0 - (now - oldest) + 0.1  # Small buffer
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def _wait_for_token_budget(self, estimated_tokens: int) -> None:
        """Wait if we've exceeded token budget."""
        if self._token_budget is None:
            return

        now = time.time()
        # Reset window if more than 60 seconds have passed
        if now - self._token_window_start >= 60.0:
            self._token_window_start = now
            self._token_window_tokens = 0

        if self._token_window_tokens + estimated_tokens > self._token_budget:
            # Wait until window resets
            wait_time = 60.0 - (now - self._token_window_start) + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._token_window_start = time.time()
                self._token_window_tokens = 0

        self._token_window_tokens += estimated_tokens

    async def _handle_429_with_backoff(
        self, attempt: int, fn: Any, *args: Any, **kwargs: Any
    ) -> JsonDict:
        """Handle 429 errors with exponential backoff."""
        backoff = min(
            self._config.initial_backoff_s * (2 ** attempt), self._config.max_backoff_s
        )
        await asyncio.sleep(backoff)
        return await fn(*args, **kwargs)

    def _should_degrade_context(self, estimated_tokens: int) -> bool:
        """Check if context should be degraded."""
        if not self._config.enable_context_degradation:
            return False
        return estimated_tokens > self._config.context_degradation_threshold

    def _degrade_context(self, messages: List[JsonDict]) -> List[JsonDict]:
        """
        Degrade context by truncating message history (keep system + last N messages).
        """
        if len(messages) <= 2:
            return messages

        # Keep system message and last 2 messages
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        last_messages = messages[-2:] if system_msg else messages[-3:]

        degraded = []
        if system_msg:
            degraded.append(system_msg)
        degraded.extend(last_messages)
        return degraded

    async def call_with_rate_limit(
        self, fn: Any, estimated_tokens: int, *args: Any, **kwargs: Any
    ) -> JsonDict:
        """
        Execute an LLM call with rate limiting, token budgeting, and error handling.
        """
        # Check if context degradation is needed
        if self._should_degrade_context(estimated_tokens):
            # Modify messages in kwargs if present
            if "messages" in kwargs:
                kwargs["messages"] = self._degrade_context(kwargs["messages"])

        async with self._semaphore:
            await self._wait_for_rate_limit()
            await self._wait_for_token_budget(estimated_tokens)

            self._request_times.append(time.time())

            # Retry with exponential backoff
            last_error = None
            for attempt in range(self._config.max_retries):
                try:
                    result = await fn(*args, **kwargs)
                    # Check for 429 in response (LiteLLM may return this in error field)
                    if isinstance(result, dict) and result.get("error"):
                        error_code = result.get("error", {}).get("code") if isinstance(result.get("error"), dict) else None
                        if error_code == 429 or "429" in str(result.get("error")):
                            if attempt < self._config.max_retries - 1:
                                result = await self._handle_429_with_backoff(attempt, fn, *args, **kwargs)
                                continue
                    return result
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                        if attempt < self._config.max_retries - 1:
                            result = await self._handle_429_with_backoff(attempt, fn, *args, **kwargs)
                            continue
                    # Re-raise if not a rate limit error or out of retries
                    if attempt == self._config.max_retries - 1:
                        raise

            if last_error:
                raise last_error
            raise RuntimeError("Rate limit retries exhausted")


class LLMGateway(Protocol):
    """
    Minimal chat gateway. Returned payload matches an OpenAI-ish response shape:
    either a tool call (name + arguments) or plain assistant content.
    """

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict: ...


class AsyncLLMGateway(Protocol):
    async def achat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict: ...


@dataclass
class LiteLLMGateway:
    """
    LiteLLM-backed gateway (optional dependency: litellm).

    Requires provider credentials via environment variables depending on provider.
    Example for OpenAI-compatible: set OPENAI_API_KEY.
    """

    api_base: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit_config: Optional[RateLimitConfig] = None

    def __post_init__(self) -> None:
        if self.rate_limit_config is not None:
            object.__setattr__(self, "_rate_limiter", RateLimiter(self.rate_limit_config))
        else:
            object.__setattr__(self, "_rate_limiter", None)

    def _kwargs(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        tool_payload = [t.as_openai_tool() for t in (tools or [])] or None
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if top_k is not None and int(top_k) > 0:
            kwargs["top_k"] = int(top_k)
        if top_p is not None:
            try:
                top_p_f = float(top_p)
                if 0.0 < top_p_f <= 1.0:
                    kwargs["top_p"] = top_p_f
            except Exception:
                pass
        if seed is not None:
            kwargs["seed"] = seed
        if self.api_base:
            kwargs["api_base"] = self.api_base
            # For OpenAI-compatible local servers (e.g. llama-server), force provider resolution.
            kwargs["custom_llm_provider"] = "openai"
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        elif self.api_base:
            # Many local OpenAI-compatible servers accept a dummy key.
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "local"
        if tool_payload is not None:
            kwargs["tools"] = tool_payload
        if tool_choice is not None:
            # OpenAI-style: "auto" or {"type":"function","function":{"name":...}}
            if tool_choice == "auto":
                kwargs["tool_choice"] = "auto"
            else:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        return kwargs

    @staticmethod
    def _should_retry_without_top_k(err: Exception) -> bool:
        s = str(err).lower()
        if "top_k" not in s:
            return False
        return any(
            x in s
            for x in (
                "unrecognized",
                "unrecognised",
                "unknown",
                "unexpected",
                "invalid",
                "not supported",
                "unsupported",
            )
        )

    @staticmethod
    def _should_retry_without_top_p(err: Exception) -> bool:
        s = str(err).lower()
        if "top_p" not in s:
            return False
        return any(
            x in s
            for x in (
                "unrecognized",
                "unrecognised",
                "unknown",
                "unexpected",
                "invalid",
                "not supported",
                "unsupported",
            )
        )

    @staticmethod
    def _strip_top_k_if_known_unsupported(*, litellm_mod: Any, kwargs: Dict[str, Any]) -> None:
        """
        Avoid sending top_k to providers that almost certainly reject it (e.g., OpenAI/Azure),
        unless an explicit api_base is set (likely a permissive local server).
        """
        if "top_k" not in kwargs:
            return
        if kwargs.get("api_base"):
            return
        try:
            provider, _model, _api_base, _api_key = litellm_mod.get_llm_provider(  # type: ignore[attr-defined]
                model=str(kwargs.get("model") or ""),
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                api_base=kwargs.get("api_base"),
                api_key=kwargs.get("api_key"),
            )
        except Exception:
            return
        if str(provider).lower() in ("openai", "azure"):
            kwargs.pop("top_k", None)

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        try:
            import litellm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LiteLLM is not installed. Install extras: `pip install -e .[cognitive]`"
            ) from e

        kwargs = self._kwargs(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        self._strip_top_k_if_known_unsupported(litellm_mod=litellm, kwargs=kwargs)

        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:
            if "top_k" in kwargs and self._should_retry_without_top_k(e):
                kwargs.pop("top_k", None)
                resp = litellm.completion(**kwargs)
                try:
                    if isinstance(resp, dict):
                        resp.setdefault("_aam_meta", {})["top_k_ignored"] = True
                except Exception:
                    pass
            elif "top_p" in kwargs and self._should_retry_without_top_p(e):
                kwargs.pop("top_p", None)
                resp = litellm.completion(**kwargs)
                try:
                    if isinstance(resp, dict):
                        resp.setdefault("_aam_meta", {})["top_p_ignored"] = True
                except Exception:
                    pass
            else:
                raise
        # Keep full response for downstream parsing.
        return resp

    async def achat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        """
        True async (preferred for Phase 4 Barrier Scheduler).
        Includes rate limiting and backpressure handling if configured.
        """
        try:
            import litellm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LiteLLM is not installed. Install extras: `pip install -e .[cognitive]`"
            ) from e

        kwargs = self._kwargs(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        self._strip_top_k_if_known_unsupported(litellm_mod=litellm, kwargs=kwargs)

        async def _call() -> JsonDict:
            # Prefer LiteLLM async API if available.
            acompletion = getattr(litellm, "acompletion", None)
            try:
                if callable(acompletion):
                    return await acompletion(**kwargs)

                # Fallback: run the sync call in a worker thread to avoid blocking the loop.
                return await asyncio.to_thread(litellm.completion, **kwargs)
            except Exception as e:
                if "top_k" in kwargs and self._should_retry_without_top_k(e):
                    kwargs2 = dict(kwargs)
                    kwargs2.pop("top_k", None)
                    if callable(acompletion):
                        resp = await acompletion(**kwargs2)
                    else:
                        resp = await asyncio.to_thread(litellm.completion, **kwargs2)
                    try:
                        if isinstance(resp, dict):
                            resp.setdefault("_aam_meta", {})["top_k_ignored"] = True
                    except Exception:
                        pass
                    return resp
                if "top_p" in kwargs and self._should_retry_without_top_p(e):
                    kwargs2 = dict(kwargs)
                    kwargs2.pop("top_p", None)
                    if callable(acompletion):
                        resp = await acompletion(**kwargs2)
                    else:
                        resp = await asyncio.to_thread(litellm.completion, **kwargs2)
                    try:
                        if isinstance(resp, dict):
                            resp.setdefault("_aam_meta", {})["top_p_ignored"] = True
                    except Exception:
                        pass
                    return resp
                raise

        # Apply rate limiting if configured
        if self._rate_limiter is not None:
            estimated_tokens = self._rate_limiter._estimate_tokens(messages, tools)
            return await self._rate_limiter.call_with_rate_limit(_call, estimated_tokens)

        return await _call()


@dataclass
class MockLLMGateway:
    """
    Deterministic offline gateway for development/testing without network access.
    It alternates between posting a message and no-op, with seeded randomness.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        _ = (top_k, top_p)
        # Fabricate a minimal OpenAI-like response structure.
        # We bias toward tool calls if tools are provided.
        do_tool = bool(tools) and (tool_choice in (None, "auto") or tool_choice == "post_message")
        if do_tool:
            # generate a short message; keep deterministic by hashing the last user content if any
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break
            content = f"hello ({abs(hash(last_user)) % 1000})"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "mock_tool_call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "post_message",
                                        "arguments": json.dumps({"content": content}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        return {"choices": [{"message": {"role": "assistant", "content": "noop"}}]}

    async def achat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        return self.chat(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )


@dataclass
class TransformerLensGateway:
    """
    Local TransformerLens-backed gateway (Phase 3).

    - Uses `transformer_lens.HookedTransformer` to generate text.
    - Does NOT support native tool calling; callers should use the text JSON path.
    - Optionally integrates with CaptureContext to record activations aligned to steps.
    """

    model_id: str
    device: Optional[str] = None
    capture_context: Optional["CaptureContext"] = None
    max_new_tokens: int = 128

    # Capability hint for the Phase 2/3 policy adapter.
    supports_tool_calls: bool = False

    def __post_init__(self) -> None:
        try:
            from transformer_lens import HookedTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TransformerLens is not installed. Install extras: `pip install -e .[interpretability]`"
            ) from e

        # Choose best available device if not explicitly provided.
        # Priority: CUDA (HPC) -> MPS (Apple Silicon) -> CPU.
        if not self.device:
            try:
                import torch  # type: ignore

                if getattr(torch.cuda, "is_available", lambda: False)():
                    self.device = "cuda"
                elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"

        # Lazy-load local model at init so subsequent calls are fast.
        # Note: downloading weights happens on the user's machine when they run phase3.
        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        
        # Try official model list first, then fall back to from_pretrained_no_processing
        # for custom models like Olmo that aren't in the official list
        try:
            self._model = HookedTransformer.from_pretrained(self.model_id, **kwargs)
        except (ValueError, KeyError) as e:
            # If model not found in official list, try loading directly from HuggingFace
            error_str = str(e).lower()
            if "not found" in error_str or "valid official model names" in error_str or "official model name" in error_str:
                print(f"Model {self.model_id} not in official TransformerLens list.")
                print("Attempting HuggingFace->TransformerLens bridge via `hf_model=`...")
                try:
                    from pathlib import Path

                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                    # Prefer repo-local cache if present (avoids re-downloads).
                    cache_dir = None
                    local_model_dir: Optional[str] = None
                    try:
                        current = Path(__file__).resolve()
                        repo_root = None
                        for parent in current.parents:
                            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                                repo_root = parent
                                break
                        if repo_root is not None:
                            candidate = repo_root / "models" / "huggingface_cache"
                            if candidate.exists():
                                cache_dir = str(candidate)
                                # Our repo stores models as plain folders like:
                                #   models/huggingface_cache/allenai_Olmo-3-1025-7B/
                                # This is not the default HF cache layout, so prefer loading directly
                                # from that folder when it exists to avoid any network fetch.
                                folder_name = self.model_id.replace("/", "_")
                                direct_path = candidate / folder_name
                                if direct_path.exists():
                                    local_model_dir = str(direct_path)
                    except Exception:
                        cache_dir = None

                    model_src = local_model_dir or self.model_id
                    local_only = bool(local_model_dir)
                    if local_only:
                        print(f"  Using local model folder: {model_src}")
                    else:
                        print("  Local model folder not found; may download from HuggingFace.")

                    tokenizer = AutoTokenizer.from_pretrained(
                        model_src,
                        cache_dir=cache_dir,
                        local_files_only=local_only,
                    )
                    import torch  # type: ignore

                    # Pick a reasonable dtype per backend.
                    # - CUDA: fp16 typically fastest
                    # - MPS: fp16 usually supported
                    # - CPU: fp32 safest
                    dev = self.device or "cpu"
                    if dev == "cuda":
                        dtype = torch.float16
                    elif dev == "mps":
                        dtype = torch.float16
                    else:
                        dtype = torch.float32

                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_src,
                        cache_dir=cache_dir,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=local_only,
                    )
                    try:
                        if dev in ("cuda", "mps", "cpu"):
                            hf_model = hf_model.to(dev)
                    except Exception:
                        pass

                    # TransformerLens insists on an "official" model_name string; use a known one
                    # and pass the real HF model+tokenizer through.
                    self._model = HookedTransformer.from_pretrained_no_processing(
                        "gpt2",
                        hf_model=hf_model,
                        tokenizer=tokenizer,
                        **kwargs,
                    )
                    print(f"✓ Successfully loaded {self.model_id} via hf_model bridge")
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load model {self.model_id} for TransformerLens.\n"
                        f"- Original error: {e}\n"
                        f"- Bridge error: {e2}"
                    ) from e2
            else:
                raise

    def _messages_to_prompt(self, messages: List[JsonDict]) -> str:
        # Minimal deterministic formatting; keep it simple for TL + JSON-output prompting.
        lines: List[str] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            lines.append(f"{role.upper()}:\n{content}\n")
        lines.append("ASSISTANT:\n")
        return "\n".join(lines)

    def on_action_decided(self, *, run_id: str, time_step: int, agent_id: str, action_name: str) -> None:
        """
        Optional post-decision callback invoked by the policy after parsing an action.
        This is where we can keep/discard buffered activations based on trigger_actions.
        """
        if self.capture_context is None:
            return
        self.capture_context.on_action_decided(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            model_id=self.model_id,
            action_name=action_name,
        )

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        # `model` is ignored for TL; we use self.model_id (keeps LLMGateway interface stable).
        _ = (model, tools, tool_choice)

        prompt = self._messages_to_prompt(messages)

        # If capture is enabled, set up hooks for this call.
        if self.capture_context is not None:
            hooks = self.capture_context.build_fwd_hooks()
            self.capture_context.begin_inference()
            with self._model.hooks(fwd_hooks=hooks):
                text = self._generate(prompt=prompt, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
        else:
            text = self._generate(prompt=prompt, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)

        # Return an OpenAI-ish response shape consumed by existing parsers.
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    async def achat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        # TL generation is compute-bound and synchronous; isolate it so the scheduler can remain async.
        return await asyncio.to_thread(
            self.chat,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

    def _generate(
        self,
        *,
        prompt: str,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        # Set seed for reproducibility if provided
        if seed is not None:
            import torch  # type: ignore
            torch.manual_seed(seed)
        # HookedTransformer.generate API varies slightly across versions; be defensive.
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if float(temperature) > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            if top_k is not None and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)
            if top_p is not None:
                try:
                    top_p_f = float(top_p)
                    if 0.0 < top_p_f <= 1.0:
                        gen_kwargs["top_p"] = top_p_f
                except Exception:
                    pass
        try:
            out = self._model.generate(prompt, **gen_kwargs)
        except TypeError:
            # Retry without sampling kwargs (older TL versions).
            gen_kwargs.pop("top_k", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("temperature", None)
            out = self._model.generate(prompt, **gen_kwargs)
        return out if isinstance(out, str) else str(out)


@dataclass
class HuggingFaceHookedGateway:
    """
    Local HF (transformers) gateway with activation capture support.

    This exists primarily for architectures not yet supported by TransformerLens
    weight-conversion (e.g. OLMo3), while still enabling the same style of
    hook names used throughout the AAM interpretability pipeline:
      - blocks.{L}.hook_resid_post

    Device priority (if not specified): CUDA -> MPS -> CPU.
    """

    model_id_or_path: str
    device: Optional[str] = None
    capture_context: Optional["CaptureContext"] = None
    max_new_tokens: int = 128
    dtype: Optional[str] = None  # "float16"|"bfloat16"|"float32"|None

    supports_tool_calls: bool = False

    def __post_init__(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HF gateway requires `transformers` + `torch` installed."
            ) from e

        # Choose best device if not explicitly provided.
        if not self.device:
            if getattr(torch.cuda, "is_available", lambda: False)():
                self.device = "cuda"
            elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
                self.device = "mps"
            else:
                self.device = "cpu"

        dev = self.device or "cpu"

        # Pick dtype
        if self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.dtype == "float32":
            torch_dtype = torch.float32
        elif self.dtype == "float16":
            torch_dtype = torch.float16
        else:
            # Default: cuda/mps -> fp16, cpu -> fp32
            torch_dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32

        # Best-effort: sanitize cached HF config.json before loading.
        #
        # Some model configs (notably OLMo3 variants) ship with integer `beta_fast`/`beta_slow`
        # under `rope_scaling` (or `rope_parameters` in some forks). Newer Transformers versions
        # warn (or may error) when these fields are not floats.
        try:
            if os.path.isdir(self.model_id_or_path):
                cfg_json_path = os.path.join(self.model_id_or_path, "config.json")
                if os.path.isfile(cfg_json_path):
                    import json

                    with open(cfg_json_path, "r") as f:
                        raw = json.load(f)
                    changed = False
                    for field in ("rope_scaling", "rope_parameters"):
                        v = raw.get(field)
                        if isinstance(v, dict):
                            for k in ("beta_fast", "beta_slow"):
                                if k in v and isinstance(v[k], int):
                                    v[k] = float(v[k])
                                    changed = True
                    if changed:
                        tmp_path = cfg_json_path + ".tmp"
                        with open(tmp_path, "w") as f:
                            json.dump(raw, f, indent=2)
                            f.write("\n")
                        os.replace(tmp_path, cfg_json_path)
        except Exception:
            pass

        # Load config and sanitize rope_* types in-memory.
        cfg = AutoConfig.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        try:
            for attr in ("rope_scaling", "rope_parameters"):
                rs = getattr(cfg, attr, None)
                if isinstance(rs, dict):
                    if "beta_fast" in rs and isinstance(rs["beta_fast"], int):
                        rs["beta_fast"] = float(rs["beta_fast"])
                    if "beta_slow" in rs and isinstance(rs["beta_slow"], int):
                        rs["beta_slow"] = float(rs["beta_slow"])
        except Exception:
            pass

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        print(f"  [HF Gateway] Loading model weights...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            config=cfg,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"  [HF Gateway] Model weights loaded")

        print(f"  [HF Gateway] Moving model to device: {dev} (this may take 30-60s for 7B models on MPS)...")
        try:
            self._model = self._model.to(dev)
            print(f"  [HF Gateway] Model moved to {dev}")
        except Exception as e:
            # Some backends rely on device_map; best-effort.
            print(f"  [HF Gateway] Warning: device transfer failed ({e}), continuing anyway")
            pass

        print(f"  [HF Gateway] Setting model to eval mode...")
        self._model.eval()
        print(f"  [HF Gateway] Model in eval mode")

        # Register hooks for requested TL-style hook names.
        self._hooks: List[Any] = []
        if self.capture_context is not None:
            print(f"  [HF Gateway] Registering activation hooks...")
            base = getattr(self._model, "model", None)
            layers = getattr(base, "layers", None) if base is not None else None
            if layers is None:
                raise RuntimeError("HF gateway could not find decoder layers at model.model.layers")

            # Determine which standardized hook names are requested from the CaptureContext config.
            try:
                requested_hook_names = [name for (name, _fn) in self.capture_context.build_fwd_hooks()]
            except Exception:
                requested_hook_names = []
            requested = set(str(h) for h in requested_hook_names)

            def _record(name: str, tensor_like: Any) -> None:
                try:
                    self.capture_context.record_activation(hook_name=name, activations=tensor_like)
                except Exception:
                    return

            num_layers = len(layers)
            for i, layer in enumerate(layers):
                # Residual pre/post captures
                resid_post_name = f"blocks.{i}.hook_resid_post"
                resid_pre_name = f"blocks.{i}.hook_resid_pre"

                if resid_post_name in requested:
                    def _make_resid_post_hook(name: str = resid_post_name):
                        def _hook(_module, _inp, out):
                            hs = out[0] if isinstance(out, (tuple, list)) else out
                            _record(name, hs)
                            return out
                        return _hook
                    self._hooks.append(layer.register_forward_hook(_make_resid_post_hook()))

                if resid_pre_name in requested:
                    def _make_resid_pre_hook(name: str = resid_pre_name):
                        def _pre_hook(_module, inp):
                            try:
                                hs_in = inp[0] if isinstance(inp, (tuple, list)) and inp else inp
                                _record(name, hs_in)
                            except Exception:
                                pass
                            return None
                        return _pre_hook
                    self._hooks.append(layer.register_forward_pre_hook(_make_resid_pre_hook()))

                # MLP output capture (best-effort)
                mlp_name = f"blocks.{i}.hook_mlp_out"
                mlp = getattr(layer, "mlp", None)
                if mlp_name in requested and mlp is not None:
                    def _make_mlp_hook(name: str = mlp_name):
                        def _hook(_module, _inp, out):
                            hs = out[0] if isinstance(out, (tuple, list)) else out
                            _record(name, hs)
                            return out
                        return _hook
                    self._hooks.append(mlp.register_forward_hook(_make_mlp_hook()))

                # Attention projection captures (best-effort; supports fused or separate q/k/v)
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    q_name = f"blocks.{i}.attn.hook_q"
                    k_name = f"blocks.{i}.attn.hook_k"
                    v_name = f"blocks.{i}.attn.hook_v"
                    result_name = f"blocks.{i}.attn.hook_result"
                    pattern_name = f"blocks.{i}.attn.hook_pattern"

                    want_q = q_name in requested
                    want_k = k_name in requested
                    want_v = v_name in requested
                    want_result = result_name in requested
                    want_pattern = pattern_name in requested

                    qkv_proj = getattr(attn, "qkv_proj", None)
                    if qkv_proj is not None and (want_q or want_k or want_v):
                        def _make_qkv_hook(
                            qn: str = q_name,
                            kn: str = k_name,
                            vn: str = v_name,
                        ):
                            def _hook(_module, _inp, out):
                                x = out[0] if isinstance(out, (tuple, list)) else out
                                try:
                                    # Expect last dim = 3 * hidden
                                    chunk = int(x.shape[-1] // 3)
                                    if want_q:
                                        _record(qn, x[..., 0 * chunk : 1 * chunk])
                                    if want_k:
                                        _record(kn, x[..., 1 * chunk : 2 * chunk])
                                    if want_v:
                                        _record(vn, x[..., 2 * chunk : 3 * chunk])
                                except Exception:
                                    pass
                                return out
                            return _hook
                        self._hooks.append(qkv_proj.register_forward_hook(_make_qkv_hook()))
                    else:
                        # Separate projections
                        if want_q and hasattr(attn, "q_proj"):
                            def _make_proj_hook(name: str = q_name):
                                def _hook(_module, _inp, out):
                                    x = out[0] if isinstance(out, (tuple, list)) else out
                                    _record(name, x)
                                    return out
                                return _hook
                            self._hooks.append(getattr(attn, "q_proj").register_forward_hook(_make_proj_hook()))
                        if want_k and hasattr(attn, "k_proj"):
                            def _make_proj_hook(name: str = k_name):
                                def _hook(_module, _inp, out):
                                    x = out[0] if isinstance(out, (tuple, list)) else out
                                    _record(name, x)
                                    return out
                                return _hook
                            self._hooks.append(getattr(attn, "k_proj").register_forward_hook(_make_proj_hook()))
                        if want_v and hasattr(attn, "v_proj"):
                            def _make_proj_hook(name: str = v_name):
                                def _hook(_module, _inp, out):
                                    x = out[0] if isinstance(out, (tuple, list)) else out
                                    _record(name, x)
                                    return out
                                return _hook
                            self._hooks.append(getattr(attn, "v_proj").register_forward_hook(_make_proj_hook()))

                    # Attention result (o_proj) capture
                    o_proj = getattr(attn, "o_proj", None)
                    if want_result and o_proj is not None:
                        def _make_o_hook(name: str = result_name):
                            def _hook(_module, _inp, out):
                                x = out[0] if isinstance(out, (tuple, list)) else out
                                _record(name, x)
                                return out
                            return _hook
                        self._hooks.append(o_proj.register_forward_hook(_make_o_hook()))
                    elif want_result:
                        # Fallback: hook the attention module output (may be (attn_out, attn_weights, ...))
                        def _make_attn_out_hook(name: str = result_name):
                            def _hook(_module, _inp, out):
                                try:
                                    x = out[0] if isinstance(out, (tuple, list)) else out
                                    _record(name, x)
                                except Exception:
                                    pass
                                return out
                            return _hook
                        self._hooks.append(attn.register_forward_hook(_make_attn_out_hook()))

                    # Attention pattern capture (best-effort)
                    if want_pattern:
                        def _make_attn_pattern_hook(name: str = pattern_name):
                            def _hook(_module, _inp, out):
                                # Attempt to locate attention weights within the output.
                                try:
                                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                                        attn_w = out[1]
                                        _record(name, attn_w)
                                except Exception:
                                    pass
                                return out
                            return _hook
                        self._hooks.append(attn.register_forward_hook(_make_attn_pattern_hook()))

                if (i + 1) % 8 == 0 or (i + 1) == num_layers:
                    print(f"  [HF Gateway] Registered hooks for {i + 1}/{num_layers} layers")
            print(f"  [HF Gateway] All {num_layers} hooks registered")
        print(f"  [HF Gateway] Initialization complete")

    def _messages_to_prompt_legacy(self, messages: List[JsonDict]) -> str:
        """Legacy prompt formatting - used as fallback when no chat template available."""
        lines: List[str] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            lines.append(f"{role.upper()}:\n{content}\n")
        lines.append("ASSISTANT:\n")
        return "\n".join(lines)

    def _messages_to_prompt(self, messages: List[JsonDict]) -> Tuple[str, bool]:
        """
        Convert messages to a prompt string, preferring chat templates.
        
        Returns:
            Tuple of (prompt_string, used_chat_template)
            - used_chat_template: True if tokenizer.apply_chat_template was used
        """
        # Try to use the tokenizer's chat template (preferred for instruct models)
        try:
            if hasattr(self._tokenizer, "apply_chat_template"):
                # Check if chat template is actually configured
                if getattr(self._tokenizer, "chat_template", None) is not None:
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return prompt, True
        except Exception as e:
            print(f"      [HF Gateway] Warning: apply_chat_template failed ({e}), using legacy format")
        
        # Fallback to legacy formatting
        return self._messages_to_prompt_legacy(messages), False

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        _ = (model, tools, tool_choice)
        prompt, used_chat_template = self._messages_to_prompt(messages)

        if self.capture_context is not None:
            self.capture_context.begin_inference()

        print(f"      [HF Gateway] Tokenizing prompt (chat_template={used_chat_template})...")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]
        try:
            dev = getattr(self._model, "device", None) or self.device or "cpu"
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass
        print(f"      [HF Gateway] Tokenization complete, input shape: {inputs['input_ids'].shape}")

        do_sample = float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": getattr(self._tokenizer, "eos_token_id", None),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            if top_k is not None and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)
            if top_p is not None:
                try:
                    top_p_f = float(top_p)
                    if 0.0 < top_p_f <= 1.0:
                        gen_kwargs["top_p"] = top_p_f
                except Exception:
                    pass

        import torch  # type: ignore

        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        print(f"      [HF Gateway] Starting generation (max_new_tokens={self.max_new_tokens}, seed={seed})...")
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        print(f"      [HF Gateway] Generation complete, output shape: {out.shape}")

        print(f"      [HF Gateway] Decoding tokens...")
        # Extract only the generated tokens (more robust than string prefix matching)
        generated_ids = out[0][input_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Fallback: if generated_ids decoding fails or is empty, try full decode with prefix stripping
        if not text:
            full_text = self._tokenizer.decode(out[0], skip_special_tokens=True)
            # Try multiple stripping approaches
            if full_text.startswith(prompt):
                text = full_text[len(prompt):].strip()
            else:
                # Try stripping by whitespace-normalized comparison
                prompt_normalized = " ".join(prompt.split())
                full_normalized = " ".join(full_text.split())
                if full_normalized.startswith(prompt_normalized):
                    # Find where the response actually starts
                    text = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else ""
                else:
                    text = full_text.strip()
        
        print(f"      [HF Gateway] Decoding complete, response length: {len(text)} chars")
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def on_action_decided(self, *, run_id: str, time_step: int, agent_id: str, action_name: str) -> None:
        """
        Mirror TransformerLensGateway hook sampling behavior when CaptureContext is used.
        """
        if self.capture_context is None:
            return
        self.capture_context.on_action_decided(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            model_id=str(self.model_id_or_path),
            action_name=action_name,
        )

    def get_unembedding_matrix(self) -> Any:
        """
        Best-effort access to the unembedding (lm_head) weight matrix.
        Returns a torch.Tensor (typically [vocab, d_model]).
        """
        try:
            head = getattr(self._model, "get_output_embeddings", None)
            if callable(head):
                emb = head()
                w = getattr(emb, "weight", None)
                if w is not None:
                    return w
        except Exception:
            pass
        try:
            lm_head = getattr(self._model, "lm_head", None)
            w = getattr(lm_head, "weight", None) if lm_head is not None else None
            if w is not None:
                return w
        except Exception:
            pass
        raise RuntimeError("Could not locate unembedding matrix (lm_head.weight / output embeddings).")

    def get_model_hash(self) -> str:
        """
        Return a hash of model weights for dual-stack verification.
        
        This allows comparing model identity between inference and probing
        to detect the "dual-stack" validity threat where different model
        weights are used for generation vs. analysis.
        
        Returns:
            16-character hex string (truncated SHA256)
        """
        import hashlib
        h = hashlib.sha256()
        # Sample weights from key layers for efficiency
        for name, param in self._model.named_parameters():
            # Sample first 1KB of each parameter to keep hashing fast
            try:
                data = param.data.cpu().numpy().tobytes()[:1000]
                h.update(name.encode("utf-8"))
                h.update(data)
            except Exception:
                continue
        return h.hexdigest()[:16]

    def register_intervention_hook(self, *, layer_idx: int, hook_fn: Any) -> Any:
        """
        Register a forward hook on decoder layer `layer_idx` and return the handle.
        Caller is responsible for removing the hook via handle.remove().
        """
        base = getattr(self._model, "model", None)
        layers = getattr(base, "layers", None) if base is not None else None
        if layers is None:
            raise RuntimeError("HF gateway could not find decoder layers at model.model.layers")
        layer = layers[int(layer_idx)]
        return layer.register_forward_hook(hook_fn)



@dataclass
class HuggingFaceTransformersGateway:
    """
    Local HuggingFace Transformers-backed gateway for local models.
    
    - Uses transformers library directly to load and run models
    - Supports local cached models
    - Does NOT support tool calling
    """
    
    model_id: str
    device: Optional[str] = None
    max_new_tokens: int = 128
    cache_dir: Optional[str] = None
    
    # Capability hint
    supports_tool_calls: bool = False
    
    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers library not installed. Install with: pip install transformers torch"
            ) from e
        
        # Lazy load model on first use
        self._model = None
        self._tokenizer = None
        self._model_class = AutoModelForCausalLM
        self._tokenizer_class = AutoTokenizer
    
    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading model {self.model_id} from cache...")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
            self._model = self._model.to(device)
        
        print(f"✓ Model loaded on {device}")
    
    def _messages_to_prompt(self, messages: List[JsonDict]) -> str:
        """Convert messages to a prompt string."""
        # Simple formatting - can be improved with chat templates
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)
    
    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        """Generate a response using the local model."""
        _ = (model, tools, tool_choice)  # Ignored for local models
        
        self._load_model()
        
        prompt = self._messages_to_prompt(messages)
        
        import torch
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device or hasattr(self._model, "device"):
            device = getattr(self._model, "device", None) or (self.device or "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        top_p_value: Optional[float] = None
        if temperature > 0 and top_p is not None:
            try:
                top_p_f = float(top_p)
                if 0.0 < top_p_f <= 1.0:
                    top_p_value = top_p_f
            except Exception:
                top_p_value = None
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_k=(int(top_k) if (temperature > 0 and top_k is not None and int(top_k) > 0) else None),
                top_p=top_p_value,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new text (after the prompt)
        if generated_text.startswith(prompt):
            response_text = generated_text[len(prompt):].strip()
        else:
            response_text = generated_text.strip()
        
        return {"choices": [{"message": {"role": "assistant", "content": response_text}}]}
    
    async def achat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> JsonDict:
        """Async version - runs in thread pool."""
        return await asyncio.to_thread(
            self.chat,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
