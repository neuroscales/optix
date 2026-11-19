# Streaming statistics (`optix.stats`)

Mergeable per-pixel summaries computed in one pass over frame batches, and exact
order statistics computed over row tiles.

::: optix.stats
    options:
      heading_level: 2
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - reduce
        - reduce_groups
        - Reducer
        - Moments
        - MinMax
        - Histogram
        - Reservoir
        - order_statistics
        - median_mad
        - tile_reduce
