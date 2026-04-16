"""
Renderer + tokenizer cache for tinker_service.

Keyed on (model_name, renderer_name). Loading a renderer is expensive (tokenizer
download, renderer init), so we LRU-cache entries per-process. Per-key locks
prevent cold-start stampede when multiple concurrent /step requests hit the
same uncached (model, renderer).

For tinker:// paths we resolve to a HuggingFace base model via the Tinker REST
API; the tokenizer is loaded for the base model, since the tinker checkpoint
shares vocab with its base.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("tinker_service.renderer_cache")


MAX_CACHE_SIZE = 20


class RendererEntry:
    def __init__(self, tokenizer: Any, renderer: Any) -> None:
        self.tokenizer = tokenizer
        self.renderer = renderer


_cache: OrderedDict[str, RendererEntry] = OrderedDict()
_entry_locks: dict[str, threading.Lock] = {}
_entry_locks_guard = threading.Lock()

_base_model_cache: dict[str, str] = {}
_resolve_locks: dict[str, threading.Lock] = {}
_resolve_locks_guard = threading.Lock()


def resolve_base_model(model_name: str) -> str:
    """Resolve a tinker:// path to its HuggingFace base model name."""
    if not model_name.startswith("tinker://"):
        return model_name

    if model_name in _base_model_cache:
        return _base_model_cache[model_name]

    with _resolve_locks_guard:
        if model_name not in _resolve_locks:
            _resolve_locks[model_name] = threading.Lock()
        lock = _resolve_locks[model_name]

    with lock:
        if model_name in _base_model_cache:
            return _base_model_cache[model_name]
        try:
            import tinker

            client = tinker.ServiceClient()
            rest = client.create_rest_client()
            run = rest.get_training_run_by_tinker_path(model_name).result()
            base = run.base_model
            if base:
                logger.info("Resolved %s -> base_model %s", model_name, base)
                _base_model_cache[model_name] = base
                return base
        except Exception as e:
            logger.warning("Failed to resolve base model for %s: %s", model_name, e)

    return model_name


def get_entry(model_name: str, renderer_name: str) -> RendererEntry:
    """Load or fetch a cached (tokenizer, renderer) pair."""
    key = f"{model_name}|{renderer_name}"
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    with _entry_locks_guard:
        if key not in _entry_locks:
            _entry_locks[key] = threading.Lock()
        lock = _entry_locks[key]

    with lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        base_model = resolve_base_model(model_name)
        logger.info(
            "Loading tokenizer for %s (from %s) with renderer %s",
            base_model, model_name, renderer_name,
        )
        tokenizer = get_tokenizer(base_model)
        renderer = get_renderer(renderer_name, tokenizer=tokenizer)
        entry = RendererEntry(tokenizer, renderer)

        _cache[key] = entry
        if len(_cache) > MAX_CACHE_SIZE:
            evicted_key, _ = _cache.popitem(last=False)
            logger.info("Evicted cached entry: %s", evicted_key)

        return entry
