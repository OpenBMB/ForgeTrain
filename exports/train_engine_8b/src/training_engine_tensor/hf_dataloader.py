"""HuggingFace dataset loader for the training engine.

Loads any HuggingFace dataset (local Parquet / Arrow / JSON / JSONL or
Hub name), tokenizes it with a HuggingFace tokenizer, packs sequences
to ``seq_length`` for causal LM training, and yields batches in the
contract consumed by ``entry.run_training``:

    {"tokens":    LongTensor   [B, S],
     "labels":    LongTensor   [B, S],
     "loss_mask": FloatTensor  [B, S]}

(``position_ids`` is NOT included — :mod:`entry` builds positions
internally from ``seq_length``.  Consumers that need explicit positions
can compute ``torch.arange(seq_length).expand(B, -1)`` themselves.)

Shuffle: each rank uses ``random.Random(seed + dp_rank)`` over its
slice of document indices.  This is statistical only — there is no
bitwise alignment with the canonical NumPy-based shuffle over packed
tokens; the loader prioritises portability over bit-equal
reproducibility.

By default the rank's slice is pre-tokenised and packed into a single
in-memory ``numpy.int32`` array at build time, so the training-loop
``next()`` performs only an ``ndarray`` view + ``torch.from_numpy``
(microseconds per call).  Set ``preload=False`` to fall back to
streaming tokenisation for datasets that do not fit in host RAM.
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

__all__ = ["build_hf_dataloader", "next_hf_batch"]


def _load_tokenizer(tokenizer_path: str) -> Any:
    """Race-safe HF tokenizer load under multi-rank torchrun.

    ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)`` writes
    Python files into ``~/.cache/huggingface/modules/.../<hash>/`` on
    first load and then ``import``-s them.  With N ranks per host racing
    on the same fs prefix, a non-rank-0 process can ``import`` a half-
    written module while rank-0 is still writing.  Serialise: every
    host's LOCAL_RANK==0 warms the cache, the others wait on a barrier,
    then load from the populated cache (no writes → no race).
    """
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


def _row_to_text(
    row: dict,
    text_fields: List[str],
    text_template: Optional[str],
) -> str:
    if text_template:
        return text_template.format(**row)
    parts = [str(row[f]) for f in text_fields if row.get(f)]
    return "\n".join(parts)


class _PackedTokenDataset(IterableDataset):
    """Tokenize + pack rows into fixed-length chunks for causal LM.

    Two modes (selected by ``preload``):
      * ``preload=True`` (default): tokenize the rank's entire slice
        upfront into one int32 ``numpy.ndarray``; ``__iter__`` only does
        ``arr[i*S : i*S + S+1]`` views + ``torch.from_numpy`` per chunk.
        Per-step main-process work is microseconds.  Memory cost is
        ``≈ avg_tok_per_doc * n_doc * 4 bytes / world_size`` — for
        GSM-8K (7.5k docs × ~250 tok) that's ~7 MB / rank, totally fine.
      * ``preload=False``: legacy on-the-fly stream — kept for very
        large datasets where pre-baking the whole slice is infeasible.
        WARNING: this mode runs the tokenizer on the main process
        inside ``next()`` and is the perf bug the preload path fixes.

    Shuffle ordering in preload mode: at the start of every epoch we
    shuffle the **packed-row** indices (each row is one seq_length+1
    chunk), so the consumption pattern is "shuffle of fixed-size
    chunks", not "shuffle of variable-length docs".  That's a tiny
    statistical difference vs the legacy doc-shuffle (boundary
    positions become deterministic within an epoch), but it preserves
    every other property (per-rank seed, strided shard, EOS between
    docs).  Bake-off step_time / loss curves are insensitive to this
    distinction at 200/1000 steps.
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        seq_length: int,
        text_fields: List[str],
        text_template: Optional[str],
        seed: int,
        rank: int,
        world_size: int,
        add_eos: bool = True,
        eos_token_id_override: Optional[int] = None,
        preload: bool = True,
        max_packed_rows: Optional[int] = None,
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
        self._preload = preload
        self._max_packed_rows = max_packed_rows
        self._eos_id = (
            eos_token_id_override
            if eos_token_id_override is not None
            else tokenizer.eos_token_id
        )

        if self._preload:
            self._packed_int32 = self._prebake_packed_array()
        else:
            self._packed_int32 = None

    # ── PRELOAD MODE: pre-tokenize + pre-pack ───────────────────────────────

    def _prebake_packed_array(self) -> np.ndarray:
        """Tokenize the rank's strided slice, pack into [N, S+1] int32.

        Runs once at construction; logs duration on rank 0.  Strict
        ``int32`` is fine because PADDED_VOCAB_SIZE = 73448 << 2³¹.

        When ``max_packed_rows`` is set the loop stops as soon as the
        rank has enough tokens for that many ``[S+1]`` chunks plus a
        small slack for the trailing partial chunk; this keeps short
        runs from paying the full-dataset tokenisation cost while the
        infinite-epoch shuffle in :meth:`_iter_preload` still produces
        every batch the training loop needs.
        """
        rank0 = self._rank == 0
        t0 = time.time()

        n = len(self._dataset)
        indices = list(range(self._rank, n, self._world_size))
        chunk_len = self._seq_length + 1
        token_budget: Optional[int] = (
            (self._max_packed_rows + 1) * chunk_len
            if self._max_packed_rows is not None
            else None
        )

        # First pass: build one big 1-D token buffer for this rank.
        if token_budget is not None:
            buf_size_hint = token_budget
        else:
            buf_size_hint = max(len(indices) * 64, 1024)  # 64 tok/doc lower bound
        buf = np.empty(buf_size_hint, dtype=np.int32)
        n_tokens = 0
        capped_early = False
        for idx in indices:
            row = self._dataset[idx]
            text = _row_to_text(row, self._text_fields, self._text_template)
            if not text.strip():
                continue
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if self._add_eos and self._eos_id is not None:
                ids = ids + [self._eos_id]
            need = n_tokens + len(ids)
            if need > buf.size:
                # Grow geometrically.
                new_size = max(buf.size * 2, need)
                grown = np.empty(new_size, dtype=np.int32)
                grown[:n_tokens] = buf[:n_tokens]
                buf = grown
            buf[n_tokens:n_tokens + len(ids)] = ids
            n_tokens += len(ids)
            if token_budget is not None and n_tokens >= token_budget:
                capped_early = True
                break

        # Second pass: reshape into [N, S+1] (drop trailing partial chunk).
        n_rows = n_tokens // chunk_len
        if n_rows == 0:
            raise RuntimeError(
                f"hf_dataloader: rank {self._rank} produced only {n_tokens} "
                f"tokens, not enough for one chunk of {chunk_len}."
            )
        packed = buf[:n_rows * chunk_len].reshape(n_rows, chunk_len).copy()

        if rank0:
            cap_note = (
                f" [capped at {self._max_packed_rows} target rows]"
                if capped_early
                else ""
            )
            print(
                f"[hf_dataloader] preload tokenize done: "
                f"{n_tokens} tokens → {n_rows} packed rows of {chunk_len} "
                f"(rank0; other ranks have similar count via strided shard) "
                f"in {time.time() - t0:.2f}s{cap_note}",
                flush=True,
            )
        return packed

    # ── ITER: preload (fast) and legacy stream (slow but flexible) ──────────

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._preload:
            yield from self._iter_preload()
        else:
            yield from self._iter_stream()

    def _iter_preload(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Fast path: yield numpy views over the pre-baked array."""
        assert self._packed_int32 is not None
        arr = self._packed_int32
        n_rows = arr.shape[0]
        S = self._seq_length
        # Cache the loss_mask tensor — every chunk has the same all-ones
        # mask, no need to allocate per yield.
        loss_mask = torch.ones(S, dtype=torch.float32)
        rng = np.random.default_rng(self._seed)
        epoch_order = np.arange(n_rows)
        while True:
            rng.shuffle(epoch_order)
            for i in epoch_order:
                row = arr[i]  # int32 view, length S+1
                # Cast to int64 once at the boundary (Megatron labels
                # path expects long).  This is the only real work left
                # in the per-step main-process path.
                tokens = torch.from_numpy(row[:-1].astype(np.int64, copy=False))
                labels = torch.from_numpy(row[1:].astype(np.int64, copy=False))
                yield {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}

    def _iter_stream(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Legacy path: on-the-fly tokenize stream (kept for fallback)."""
        import random

        n = len(self._dataset)
        indices = list(range(self._rank, n, self._world_size))
        rng = random.Random(self._seed)

        def _token_stream() -> Iterator[int]:
            while True:
                rng.shuffle(indices)
                for idx in indices:
                    row = self._dataset[idx]
                    text = _row_to_text(row, self._text_fields, self._text_template)
                    if not text.strip():
                        continue
                    token_ids = self._tokenizer.encode(text, add_special_tokens=False)
                    yield from token_ids
                    if self._add_eos and self._eos_id is not None:
                        yield self._eos_id

        buf_len = self._seq_length + 1
        stream = _token_stream()
        while True:
            chunk = list(itertools.islice(stream, buf_len))
            if len(chunk) < buf_len:
                break
            tokens = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            loss_mask = torch.ones(self._seq_length, dtype=torch.float32)
            yield {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
    }


def build_hf_dataloader(
    *,
    dataset_path: str,
    tokenizer_path: str,
    micro_batch_size: int,
    seq_length: int,
    seed: int,
    dp_rank: int,
    dp_size: int,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    data_format: Optional[str] = None,
    data_files: Optional[Any] = None,
    text_fields: Optional[List[str]] = None,
    text_template: Optional[str] = None,
    eos_token_id_override: Optional[int] = None,
    num_workers: int = 0,
    preload: Optional[bool] = None,
    max_packed_rows: Optional[int] = None,
) -> DataLoader:
    """Build a streaming HF DataLoader sharded across DP ranks.

    Args:
        dataset_path: Local dir or Hub name passed to ``load_dataset``.
            Ignored when ``data_format`` is set.
        tokenizer_path: Local dir or Hub name for ``AutoTokenizer``.
        micro_batch_size: Per-GPU MBS (Megatron's ``--micro-batch-size``).
        seq_length: Packed chunk length (Megatron's ``--seq-length``).
        seed: Base shuffle seed; per-rank seed is ``seed + dp_rank``.
        dp_rank, dp_size: Data-parallel topology.
        dataset_config: e.g. ``"main"`` for GSM-8K.
        dataset_split: e.g. ``"train"``.
        data_format: Optional explicit loader name passed as the first
            positional arg of ``load_dataset`` (e.g. ``"parquet"`` /
            ``"json"``).  When set, ``data_files`` MUST be supplied and
            ``dataset_path`` is ignored.  Useful for pointing at a glob
            of parquet/json shards on a local filesystem without needing
            an HF dataset script.
        data_files: Glob string / list / dict passed to ``load_dataset``
            (only when ``data_format`` is set).
        text_fields: Columns to concat; auto-required if no template.
        text_template: Format string applied to row dict (preferred for
            structured datasets like GSM-8K's ``question``/``answer``).
        num_workers: DataLoader workers (0 = main process; safer with
            random.Random + IterableDataset).  With ``preload=True`` the
            per-step main-process cost is microseconds, so num_workers=0
            is fine (and avoids fork pitfalls with IterableDataset).
        preload: If True (default), pre-tokenize + pre-pack the rank's
            slice at build time so per-step work is just numpy view +
            torch.from_numpy.  Set to False to keep the legacy on-the-fly
            tokenize path (useful for very large datasets).
        max_packed_rows: When ``preload=True``, stop pre-tokenisation
            once the rank has produced this many ``[seq_length+1]``
            chunks (the iterator wraps with re-shuffle, so this is a
            lower bound on the rows actually consumed).  ``None`` means
            "tokenise every doc in the rank's slice" (legacy behaviour;
            wasteful for short runs on huge corpora).  Production
            callers should pass
            ``num_steps * grad_accum_steps * micro_batch_size * SAFETY``.
    """
    from datasets import load_dataset

    if data_format is not None:
        if data_files is None:
            raise ValueError(
                "build_hf_dataloader: data_format set without data_files; "
                "pass data_files=<glob/list> alongside data_format."
            )
        ds = load_dataset(
            data_format,
            data_files=data_files,
            split=dataset_split,
        )
    else:
        ds = load_dataset(
            dataset_path,
            name=dataset_config,
            split=dataset_split,
        )

    tokenizer = _load_tokenizer(tokenizer_path)

    if text_fields is None and text_template is None:
        raise ValueError(
            "build_hf_dataloader requires either text_fields=[...] or "
            "text_template='...'"
        )
    if text_fields is None:
        # Template formats from row dict directly; no field list needed.
        text_fields = []

    eos_id_effective = (
        eos_token_id_override
        if eos_token_id_override is not None
        else tokenizer.eos_token_id
    )

    if preload is None:
        preload = True

    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        if data_format is not None:
            src_desc = f"format={data_format} data_files={data_files}"
        else:
            src_desc = f"dataset={dataset_path} config={dataset_config}"
        print(
            f"[hf_dataloader] {src_desc} "
            f"split={dataset_split} rows={len(ds)}\n"
            f"[hf_dataloader] tokenizer={tokenizer_path} "
            f"len(t)={len(tokenizer)} eos_token_id(default)={tokenizer.eos_token_id} "
            f"eos_id(effective)={eos_id_effective} "
            f"override={eos_token_id_override}\n"
            f"[hf_dataloader] seq_length={seq_length} MBS={micro_batch_size} "
            f"dp_rank={dp_rank}/{dp_size} seed_base={seed} "
            f"per_rank_seed={seed + dp_rank}\n"
            f"[hf_dataloader] text_template={text_template!r} "
            f"text_fields={text_fields}\n"
            f"[hf_dataloader] preload={preload}  num_workers={num_workers}",
            flush=True,
        )

    packed = _PackedTokenDataset(
        dataset=ds,
        tokenizer=tokenizer,
        seq_length=seq_length,
        text_fields=text_fields,
        text_template=text_template,
        seed=seed + dp_rank,
        rank=dp_rank,
        world_size=dp_size,
        eos_token_id_override=eos_token_id_override,
        preload=preload,
        max_packed_rows=max_packed_rows,
    )
    return DataLoader(
        packed,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )


def next_hf_batch(
    dl_iter: Iterator[Dict[str, torch.Tensor]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Pull the next batch and move tensors to ``device``.

    Parallels ``dataloader.next_nonempty_batch`` but without:
      * the sampler-state update (HF stream has no checkpoint contract
        in this experiment);
      * the empty-batch skip loop (HF packing never emits an all-zero
        loss_mask — every packed sequence has seq_length real tokens).
    """
    data = next(dl_iter)
    return {
        "tokens":    data["tokens"].to(device, non_blocking=True),
        "labels":    data["labels"].to(device, non_blocking=True),
        "loss_mask": data["loss_mask"].to(device, non_blocking=True),
    }
