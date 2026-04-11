# FAQ

## Should I build a landing page or a documentation site?

For STEER, a documentation-first site is usually the better default. Users mainly need installation steps, tutorials, preprocessing guidance, and citation information.

## Why use Read the Docs?

Read the Docs can build your docs automatically from your Git repository using a `.readthedocs.yaml` file stored at the top level of the repo.[^rtd]

## Why use Material for MkDocs?

Read the Docs specifically recommends Material for MkDocs for MkDocs-based projects, and it gives you a clean documentation style similar to many modern scientific tools.[^mkdocs]

## Anything I should pin?

Yes. Material for MkDocs published guidance in February 2026 saying that MkDocs 2.0 is not backward-compatible with Material for MkDocs. This starter therefore pins `mkdocs<2` in `requirements-docs.txt` to avoid surprise breakage for now.[^mkdocs2]

## Where should the files live?

MkDocs expects a `mkdocs.yml` file and a top-level `docs/` directory by default.[^mkdocs_docs]
Read the Docs expects `.readthedocs.yaml` at the top level of the repository.[^rtd]

[^rtd]: Read the Docs says `.readthedocs.yaml` belongs at the top level of the Git repository and is used to configure builds. citeturn581664view1turn373744search13
[^mkdocs]: Read the Docs recommends Material for MkDocs when using MkDocs. citeturn581664view0turn373744search0
[^mkdocs2]: Material for MkDocs stated on February 18, 2026, that MkDocs 2.0 is incompatible with Material for MkDocs. citeturn581664view3
[^mkdocs_docs]: MkDocs documents the standard layout as `mkdocs.yml` plus a top-level `docs/` directory. citeturn373744search18
