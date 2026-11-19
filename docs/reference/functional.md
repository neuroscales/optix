# Distribution kernels (`optix.distributions.functional`)

Pure tensor functions behind the distribution classes. They take the canonical
parameters of each family, validate nothing and never synchronise with the
host, so they can be inlined in a compiled kernel.

::: optix.distributions.functional
    options:
      heading_level: 2
      show_root_heading: false
      show_root_toc_entry: false
