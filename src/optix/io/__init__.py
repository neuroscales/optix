"""Lazy, chunked access to array-likes (Tensor, numpy, memmap, zarr, ...).

Data are laid out as ``(*batch, frames, *spatial)``; functions take
``ndim`` = number of trailing spatial dims.
"""

from ._arrays import (
    ArrayLike,
    Index,
    WritableArrayLike,
    as_tensor,
    chunks_of,
    flush,
    prepare_out,
    store,
)
from ._iter import Chunks, blocks, chunk_length, frame_batches, row_tiles

__all__ = [
    "ArrayLike",
    "Chunks",
    "Index",
    "WritableArrayLike",
    "as_tensor",
    "blocks",
    "chunk_length",
    "chunks_of",
    "flush",
    "frame_batches",
    "prepare_out",
    "row_tiles",
    "store",
]
