# quack (vendored)

This directory contains a **pruned vendored copy** of the upstream
[`quack`](https://github.com/Dao-AILab/quack) project (Wentao Guo,
Ted Zadouri, Tri Dao). Only the helpers required by
`flash_attn_dsl` (the in-house Hopper SM90a flash-attention forward
kernel this engine ships) are included here.

## Upstream provenance

- Project: `quack` (Dao-AILab)
- Vendored snapshot version: `0.3.11-vendored` (see `__init__.py`)
- Per-file copyright headers (e.g. `Copyright (c) 2025, Wentao Guo, Ted
  Zadouri, Tri Dao.` in `utils.py`) are preserved from upstream.

## License

The upstream `quack` license has not yet been mirrored into this directory.
Before publishing this repository, replace this section with the upstream
license text (or a `LICENSE` file alongside this `NOTICE.md`) to remain
compliant with the upstream terms.

## Why vendored

`quack` is required by `flash_attn_dsl/` for SM90 CuTeDSL helper
utilities (smem layout / TMA copy / dropout / packed-pair softmax) and is
not yet installable as a stable PyPI wheel; vendoring the exact snapshot
we tested against keeps the training engine reproducible.
