# Vendored from the upstream `quack` project (Wentao Guo, Ted Zadouri,
# Tri Dao; see per-file copyright headers under this directory).
#
# This is a pruned copy carrying only the helpers that `flash_attn_dsl`
# (the in-house Hopper SM90a flash-attention forward kernel) depends on;
# it is *not* the full upstream package. See NOTICE.md in this directory
# for the upstream provenance and license placeholder.

__version__ = "0.3.11-vendored"

import os

# NVIDIA/cutlass#3161: fix duplicate .text section flags in CuTeDSL-emitted .o
# files before anything else imports / compiles. Must run before the first
# cute.compile call; see quack.cute_dsl_elf_fix for details.
try:
    import quack.cute_dsl_elf_fix
    quack.cute_dsl_elf_fix.patch()
except Exception:
    pass
