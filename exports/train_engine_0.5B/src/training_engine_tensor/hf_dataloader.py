"""HuggingFace dataset loader — drop-in alternative to modelbest_sdk.

Loads any HuggingFace dataset (local or Hub), tokenizes with a
HuggingFace tokenizer, packs sequences to ``seq_length`` for causal LM
training, and yields batches in the same contract as ``dataloader.py``:

    {"tokens": LongTensor, "labels": LongTensor, "loss_mask": FloatTensor}

Supports:
  - Local dataset directories (Parquet / Arrow / JSON / JSONL)
  - Hub dataset names (e.g. ``openai/gsm8k``)
  - Automatic text field detection or explicit ``--hf-text-field``
  - Distributed sharding via ``rank`` / ``world_size``
  - Deterministic seeding for reproducibility
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

__all__ = ["build_hf_dataloader"]


def _load_tokenizer(tokenizer_path: str) -> Any:
    """Load HF tokenizer with ``trust_remote_code``, race-safe under multi-rank.

    ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)`` writes Python
    files into ``~/.cache/huggingface/modules/transformers_modules/<repo>/<hash>/``
    on first load and then ``import``-s them. With N ranks per host racing on
    the same fs prefix, a non-rank-0 process can ``import`` a half-written
    module while rank-0 is still writing, surfacing as
    ``AttributeError: module 'transformers_modules.<repo>.<hash>.<file>'
    has no attribute '<XxxConfig>'`` (observed under multi-host PyTorchJob).

    Serialise the first load **per host**: every host's ``LOCAL_RANK==0``
    warms the cache, a global ``barrier`` waits until each host's warmer has
    finished, then the remaining ranks ``from_pretrained`` against an already
    populated cache (no writes → no race). Each host has its own warmer
    because the cache lives on host-local fs.
    """
    import os

    from transformers import AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    if use_dist:
        if local_rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True,
            )
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _detect_text_fields(dataset: Any) -> list[str]:
    """Auto-detect text columns from the dataset schema."""
    col_names = dataset.column_names
    if isinstance(col_names, dict):
        col_names = list(col_names.values())[0]

    candidates = ["text", "content", "input", "sentence", "document"]
    found = [c for c in candidates if c in col_names]
    if found:
        return found[:1]

    string_cols = []
    features = dataset.features if hasattr(dataset, "features") else None
    if features is None and hasattr(dataset, "info"):
        features = dataset.info.features
    if features is not None:
        from datasets import Value

        for name, feat in features.items():
            if isinstance(feat, Value) and feat.dtype == "string":
                string_cols.append(name)

    if string_cols:
        return string_cols
    raise ValueError(
        f"Cannot auto-detect text fields from columns: {col_names}. "
        f"Use --hf-text-field to specify explicitly."
    )


def _row_to_text(row: dict, text_fields: list[str], text_template: str | None) -> str:
    """Convert a dataset row to a single text string."""
    if text_template:
        return text_template.format(**row)
    parts = [str(row[f]) for f in text_fields if row.get(f)]
    return "\n".join(parts)


class PackedTokenDataset(IterableDataset):
    """Tokenizes + packs text into fixed-length chunks for causal LM.

    Each yielded sample is a dict with:
      - tokens: (seq_length,) int64
      - labels: (seq_length,) int64 — shifted by 1 position
      - loss_mask: (seq_length,) float32
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        seq_length: int,
        text_fields: list[str],
        text_template: str | None = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        *,
        add_eos: bool = True,
    ):
        super().__init__()
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._seq_length = seq_length
        self._text_fields = text_fields
        self._text_template = text_template
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        self._add_eos = add_eos
        self._eos_id = tokenizer.eos_token_id

    def _token_stream(self) -> Iterator[int]:
        """Yield a stream of token IDs from sharded dataset rows."""
        dataset = self._dataset
        n = len(dataset)
        indices = list(range(self._rank, n, self._world_size))

        import random

        rng = random.Random(self._seed)

        while True:
            rng.shuffle(indices)
            for idx in indices:
                row = dataset[idx]
                text = _row_to_text(row, self._text_fields, self._text_template)
                if not text.strip():
                    continue
                token_ids = self._tokenizer.encode(text, add_special_tokens=False)
                yield from token_ids
                if self._add_eos and self._eos_id is not None:
                    yield self._eos_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf_len = self._seq_length + 1
        stream = self._token_stream()

        while True:
            chunk = list(itertools.islice(stream, buf_len))
            if len(chunk) < buf_len:
                break

            tokens = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            loss_mask = torch.ones(self._seq_length, dtype=torch.float32)

            yield {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}


def _collate_packed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
    }


def build_hf_dataloader(
    dataset_path: str,
    tokenizer_path: str,
    micro_batch_size: int,
    seq_length: int,
    seed: int = 42,
    dp_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    text_fields: list[str] | None = None,
    text_template: str | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader from a HuggingFace dataset.

    Args:
        dataset_path: Local path or Hub name of the dataset.
        tokenizer_path: Local path or Hub name of the tokenizer.
        micro_batch_size: Per-GPU micro batch size.
        seq_length: Sequence length for packed samples.
        seed: Random seed for shuffling.
        dp_rank: Data-parallel rank.
        world_size: Data-parallel world size.
        num_workers: DataLoader workers (0 = main process).
        dataset_config: Dataset config/subset name (e.g. "main").
        dataset_split: Dataset split (default "train").
        text_fields: Columns to extract text from. Auto-detected if None.
        text_template: Python format string to combine columns, e.g.
            "Question: {question}\\nAnswer: {answer}".

    Returns:
        A PyTorch DataLoader yielding dicts with tokens/labels/loss_mask.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset_path,
        name=dataset_config,
        split=dataset_split,
    )

    tokenizer = _load_tokenizer(tokenizer_path)

    if text_fields is None:
        text_fields = _detect_text_fields(ds)

    print(
        f"[hf_dataloader] dataset={dataset_path} config={dataset_config} "
        f"split={dataset_split}\n"
        f"[hf_dataloader] text_fields={text_fields} "
        f"rows={len(ds)} seq_len={seq_length} MBS={micro_batch_size}\n"
        f"[hf_dataloader] tokenizer={tokenizer_path} "
        f"vocab_size={len(tokenizer)} "
        f"rank={dp_rank}/{world_size}",
        flush=True,
    )

    packed_ds = PackedTokenDataset(
        dataset=ds,
        tokenizer=tokenizer,
        seq_length=seq_length,
        text_fields=text_fields,
        text_template=text_template,
        seed=seed + dp_rank,
        rank=dp_rank,
        world_size=world_size,
    )

    return DataLoader(
        packed_ds,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=_collate_packed,
        pin_memory=True,
    )
