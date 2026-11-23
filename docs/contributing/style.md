# Style guide

How to write documentation and docstrings for optix. The guide is short on
purpose: when in doubt, follow the nearest existing page.

## References

Cite publications in a `references` admonition, in one citation style:

```text
Author A, Author B. *Title.* Journal **vol**(issue):pages, year. [doi:…](https://doi.org/…)
```

Give every reference its DOI, checked against the publisher's page. End the block
with a BibTeX entry in a collapsed `??? quote "BibTeX"` admonition. The source
looks like this (indented four spaces under the admonition):

````markdown
!!! references "References"

    1. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
       Poissonian-Gaussian noise modeling and fitting for single-image raw-data.*
       IEEE Trans Image Process **17**(10):1737-1754, 2008.
       [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

    ??? quote "BibTeX"

        ```bibtex
        @article{foi2008practical,
          ...
        }
        ```
````

and renders as:

!!! references "References"

    1. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
       Poissonian-Gaussian noise modeling and fitting for single-image raw-data.*
       IEEE Trans Image Process **17**(10):1737-1754, 2008.
       [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

    ??? quote "BibTeX"

        ```bibtex
        @article{foi2008practical,
          author  = {Foi, Alessandro and Trimeche, Mejdi and Katkovnik, Vladimir and Egiazarian, Karen},
          title   = {Practical {P}oissonian-{G}aussian noise modeling and fitting for single-image raw-data},
          journal = {IEEE Transactions on Image Processing},
          volume  = {17},
          number  = {10},
          pages   = {1737--1754},
          year    = {2008},
          doi     = {10.1109/TIP.2008.2001399},
        }
        ```

The icon and colour of the admonition are defined in `docs/stylesheets/extra.css`.

## Docs pages

- Maths is LaTeX, between `$...$` or `$$...$$`. Symbols follow the
  [notation table](../concepts/notation.md); add a symbol there when a page
  introduces one.
- Examples are `pycon` blocks that run as written, and their output is checked.
- Use admonitions (`!!! note`, `!!! warning`, `??? tip` for collapsed ones) for
  asides, sparingly.

## Docstrings

Docstrings use the numpy format and are rendered by mkdocstrings.

- **Maths in UTF-8, not LaTeX.** Write `σ² = g²λ + …` in a docstring, since it is
  read in editors and in `help()` as often as on the web. LaTeX is for docs pages
  only.
- **Cross-references use mkdocstrings syntax**, never Sphinx roles: `` [`name`][] ``
  for a name in scope, or `` [`name`][optix.x.Name] `` with the full path. Do not
  write `` :class:`Name` `` or `` :func:`name` ``.
- **Admonitions** are zensical ones (`!!! note "Title"` followed by an indented
  body), not reST directives.
- **Constructor parameters are documented in `__init__`**, not in the class
  docstring. The site is built with `merge_init_into_class = true`, so they appear
  on the class page.
- **The data argument is called `data`**, not `Y`, in every public signature.

## The API reference

There is one page per public module under `docs/reference/`, written as
`::: optix.module` blocks and listed in the navigation. A new public name is
added to the `members` list of its page. The build must pass without warnings
about unresolved cross-references.
