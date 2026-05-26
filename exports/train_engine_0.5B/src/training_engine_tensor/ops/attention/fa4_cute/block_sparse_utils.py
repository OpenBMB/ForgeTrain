"""Dense spec: block-sparse helpers are not used (compile-time const_expr guards)."""


def get_total_q_block_count_bwd(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")


def produce_block_sparse_q_loads_bwd_sm90(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")


def consume_block_sparse_mma_bwd_sm90(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")


def dQaccum_store_block_sparse_bwd_sm90(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")


def produce_block_sparse_loads(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")


def consume_block_sparse_loads(*args, **kwargs):
    raise RuntimeError("block sparsity disabled in fa4 spec build")
