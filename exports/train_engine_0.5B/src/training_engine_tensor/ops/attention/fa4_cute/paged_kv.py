"""Minimal stub for paged KV — only needed so flash_fwd_sm90.py can import PagedKVManager.

We never use paged KV for our training shapes, so nothing here is actually called.
"""


class PagedKVManager:
    """Placeholder — real implementation lives in the upstream flash_attn package."""

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError("Paged KV is not supported in this spec build")
