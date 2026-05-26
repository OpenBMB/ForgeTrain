"""JIT compile cache (minimal local stub; mirrors flash_attn.cute.cache_utils API)."""

_JIT_CACHES: dict[str, dict] = {}


def get_jit_cache(name: str) -> dict:
    if name not in _JIT_CACHES:
        _JIT_CACHES[name] = {}
    return _JIT_CACHES[name]
