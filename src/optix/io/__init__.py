"""Lazy, chunked access to array-likes (Tensor, numpy, memmap, zarr, ...).

Axis roles (spatial, sample, group) are declared with a
[`Layout`][optix.io.Layout], either attached to the data with
[`labeled`][optix.io.labeled] or passed as `layout=`. Without one, data
are `(*samples, *spatial)` with `ndim` trailing spatial dims.
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
from ._iter import (
    Chunk,
    Chunks,
    blocks,
    chunk_length,
    frame_batches,
    row_tiles,
)
from ._layout import Labeled, Layout, labeled

__all__ = [
    "ArrayLike",
    "Chunk",
    "Chunks",
    "Index",
    "Labeled",
    "Layout",
    "WritableArrayLike",
    "as_tensor",
    "blocks",
    "chunk_length",
    "chunks_of",
    "flush",
    "frame_batches",
    "labeled",
    "prepare_out",
    "row_tiles",
    "store",
]
