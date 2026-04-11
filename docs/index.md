# STEER

**STEER** is a graph-attention-based **Spatial-Temporal Explainable Expert** framework for **RNA velocity inference**. It leverages spatial-temporal gene expression information and provides modules for training, visualization, prior construction, and utilities for single-cell and spatial dynamics.[^repo]

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quick start**

    ---

    Install STEER and run the core workflow.

    [Open quick start](quickstart.md)

-   :material-book-open-page-variant: **Tutorials**

    ---

    Follow end-to-end examples and platform-specific preprocessing notes.

    [Browse tutorials](tutorials/index.md)

-   :material-github: **Source code**

    ---

    View the repository, issues, and release history.

    [Open GitHub](https://github.com/lzygenomics/STEER)

-   :material-format-quote-close: **Citation**

    ---

    Use the published reference when citing STEER in your work.

    [See citation](citation.md)

</div>

## Why STEER?

STEER is intended for researchers who want more than a black-box velocity estimate. The current repository presents it as an interpretable deep-learning framework that combines spatial-temporal structure with graph attention for RNA velocity inference.[^repo]

### Highlights

- **Interpretable modeling** for RNA velocity inference.
- **Spatially aware workflows** for modern transcriptomics data.
- **Tutorial-first organization** including a core pipeline and multiple preprocessing routes.[^tutorials]
- **Research-ready citation information** with a DOI already available.[^citation]

## Documentation map

This starter site is organized to match how most users approach a scientific software package:

1. **Installation** — set up Python, PyTorch, PyG, and R dependencies.
2. **Quick Start** — run the main STEER workflow on prepared data.
3. **Tutorials** — choose a platform-specific preprocessing route when your data does not yet contain `spliced` and `unspliced` layers.
4. **Citation** — copy the manuscript reference and DOI.

## What you should customize next

This scaffold is ready to publish, but you should replace placeholders with project-specific detail where possible:

- add screenshots or figures from your paper
- add exact notebook links for each tutorial page
- add one minimal reproducible example dataset
- add output plots to show what STEER produces
- add a changelog once releases become frequent

## Contact

Questions, issues, or collaboration requests can go through the GitHub issue tracker or the contact email listed in the repository: `lzy_math@163.com`.[^repo]

[^repo]: The STEER README describes the method, install flow, tutorials, and contact details. citeturn259328view0turn259328view3
[^tutorials]: The repository currently lists a core demo plus Slide-seq, 10x Visium, and Stereo-seq preprocessing pipelines. citeturn259328view1
[^citation]: The README includes the published citation and DOI. citeturn259328view2turn259328view4
