from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

from aam.tools import ToolSpec


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
    ) -> Dict[str, Any]:
        tool_payload = [t.as_openai_tool() for t in (tools or [])] or None
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
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

    def chat(
        self,
        *,
        model: str,
        messages: List[JsonDict],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
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
        )

        resp = litellm.completion(**kwargs)
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
        )

        async def _call() -> JsonDict:
            # Prefer LiteLLM async API if available.
            acompletion = getattr(litellm, "acompletion", None)
            if callable(acompletion):
                return await acompletion(**kwargs)

            # Fallback: run the sync call in a worker thread to avoid blocking the loop.
            return await asyncio.to_thread(litellm.completion, **kwargs)

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
    ) -> JsonDict:
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
    ) -> JsonDict:
        return self.chat(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
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

        # Lazy-load local model at init so subsequent calls are fast.
        # Note: downloading weights happens on the user's machine when they run phase3.
        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = HookedTransformer.from_pretrained(self.model_id, **kwargs)

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
    ) -> JsonDict:
        # `model` is ignored for TL; we use self.model_id (keeps LLMGateway interface stable).
        _ = (model, tools, tool_choice)

        prompt = self._messages_to_prompt(messages)

        # If capture is enabled, set up hooks for this call.
        if self.capture_context is not None:
            hooks = self.capture_context.build_fwd_hooks()
            self.capture_context.begin_inference()
            with self._model.hooks(fwd_hooks=hooks):
                text = self._generate(prompt=prompt, temperature=temperature)
        else:
            text = self._generate(prompt=prompt, temperature=temperature)

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
    ) -> JsonDict:
        # TL generation is compute-bound and synchronous; isolate it so the scheduler can remain async.
        return await asyncio.to_thread(
            self.chat,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

    def _generate(self, *, prompt: str, temperature: float) -> str:
        # HookedTransformer.generate API varies slightly across versions; be defensive.
        try:
            out = self._model.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=float(temperature),
            )
        except TypeError:
            out = self._model.generate(prompt, max_new_tokens=self.max_new_tokens)
        return out if isinstance(out, str) else str(out)


