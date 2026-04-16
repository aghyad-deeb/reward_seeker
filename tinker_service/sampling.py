"""
Sampling adapters for tinker_service.

Two backends, picked by base_url scheme:
  tinker://   -> Tinker SDK (SamplingClient.sample_async, token-level I/O)
  http(s)://  -> POST /v1/completions with prompt=<token_ids>; response text
                 re-tokenized locally (keeping harmony special tokens intact).

Both return the response as a `list[int]` of token IDs, so the caller can feed
them straight into renderer.parse_response.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import httpx

logger = logging.getLogger("tinker_service.sampling")


DEFAULT_TINKER_BASE = "https://api.tinker.thinkingmachines.com"


# ── Tinker SDK sampling client cache ───────────────────────────────────────

_sampling_clients: OrderedDict[str, Any] = OrderedDict()
_sampling_client_locks: dict[str, threading.Lock] = {}
_sampling_client_locks_guard = threading.Lock()
MAX_SAMPLING_CLIENT_CACHE = 20


def _get_sampling_client(model_name: str) -> Any | None:
    if not model_name.startswith("tinker://"):
        return None

    if model_name in _sampling_clients:
        _sampling_clients.move_to_end(model_name)
        return _sampling_clients[model_name]

    with _sampling_client_locks_guard:
        if model_name not in _sampling_client_locks:
            _sampling_client_locks[model_name] = threading.Lock()
        lock = _sampling_client_locks[model_name]

    with lock:
        if model_name in _sampling_clients:
            _sampling_clients.move_to_end(model_name)
            return _sampling_clients[model_name]
        try:
            import tinker

            sc = tinker.ServiceClient()
            client = sc.create_sampling_client(model_path=model_name)
            logger.info("Created SamplingClient for %s", model_name)
            _sampling_clients[model_name] = client
            if len(_sampling_clients) > MAX_SAMPLING_CLIENT_CACHE:
                evicted_key, _ = _sampling_clients.popitem(last=False)
                logger.info("Evicted SamplingClient: %s", evicted_key)
            return client
        except Exception as e:
            logger.warning("Failed to create SamplingClient for %s: %s", model_name, e)
            return None


# ── Public API ──────────────────────────────────────────────────────────────


async def sample(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    renderer: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    seed: int | None = None,
) -> tuple[list[int], str]:
    """Sample one completion. Returns (response_tokens, stop_reason)."""
    stop_condition = renderer.get_stop_sequences()

    if model_name.startswith("tinker://"):
        return await _sample_tinker_sdk(
            model_name=model_name,
            renderer=renderer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stop_condition=stop_condition,
        )
    return await _sample_http(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        stop_condition=stop_condition,
    )


async def _sample_tinker_sdk(
    *,
    model_name: str,
    renderer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    seed: int | None,
    stop_condition: Any,
) -> tuple[list[int], str]:
    import tinker

    client = _get_sampling_client(model_name)
    if client is None:
        raise RuntimeError(f"Could not create Tinker SamplingClient for {model_name}")

    # Tinker SDK takes a ModelInput, not a raw token list. Rebuild from the
    # tokens we already have; this is zero-cost since ModelInput is just a
    # wrapper over int[].
    model_input = tinker.ModelInput.from_ints(prompt_tokens)

    sp_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_condition,
    }
    if seed is not None:
        sp_kwargs["seed"] = seed

    result = await client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(**sp_kwargs),
    )
    seq = result.sequences[0]
    tokens = list(seq.tokens)
    stop_reason = getattr(seq, "stop_reason", "stop")
    return tokens, str(stop_reason)


async def _sample_http(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    seed: int | None,
    stop_condition: Any,
) -> tuple[list[int], str]:
    """POST /v1/completions with token-list prompt; re-tokenize text response.

    vLLM accepts `prompt` as list[int], which sidesteps the input-side
    detokenization that mangles harmony control tokens. For the response side,
    we re-encode the returned text with skip_special_tokens=False so those
    control bytes survive the round-trip.
    """
    url = (base_url or os.environ.get("TINKER_BASE_URL") or DEFAULT_TINKER_BASE).rstrip("/")
    key = api_key or os.environ.get("TINKER_API_KEY") or ""

    stop_strs = _stop_condition_to_strings(tokenizer, stop_condition)

    body: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_strs,
        # vLLM extension: keep special tokens in the returned text so we can
        # re-tokenize the harmony control bytes.
        "skip_special_tokens": False,
    }
    if seed is not None:
        body["seed"] = seed

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        resp = await client.post(f"{url}/v1/completions", json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Completions API returned {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()

    choice = (data.get("choices") or [{}])[0]
    text = choice.get("text", "")
    stop_reason = choice.get("finish_reason", "stop") or "stop"

    if not text:
        return [], str(stop_reason)

    # Re-tokenize without forcing BOS/EOS so control tokens stay in their
    # natural positions. add_special_tokens=False is the conservative choice;
    # the renderer's parse_response handles stop-token tails explicitly.
    try:
        response_tokens = list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        response_tokens = list(tokenizer.encode(text))
    return response_tokens, str(stop_reason)


def _stop_condition_to_strings(tokenizer: Any, stop_condition: Any) -> list[str]:
    """Coerce a renderer's stop condition (list[int] | list[str]) into strings
    for the OpenAI-compat /completions `stop` field.
    """
    if stop_condition is None:
        return []
    out: list[str] = []
    for s in stop_condition:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, int):
            out.append(tokenizer.decode([s]))
        elif isinstance(s, (list, tuple)) and s:
            out.append(tokenizer.decode(list(s)))
    # OpenAI/vLLM caps `stop` at 4 entries; dedup and trim.
    seen: set[str] = set()
    dedup: list[str] = []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup[:4]
