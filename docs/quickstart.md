# Quick Start

This page is the shortest path from installation to a first successful run.

## Before you begin

STEER can be run directly when your input `.h5ad` already contains **`spliced`** and **`unspliced`** layers. If not, start from one of the preprocessing tutorials instead.[^tutorials]

## Typical workflow

```python
# Example skeleton only — replace with your real notebook commands.
import scanpy as sc

adata = sc.read_h5ad("your_data.h5ad")

# TODO: add the exact STEER API calls used in your notebook
# model = ...
# result = ...
# result.plot(...)
```

## Suggested quick-start structure

Use this page to document the minimum reproducible workflow:

1. load a prepared `.h5ad`
2. initialize the STEER model
3. train or infer velocity
4. save outputs
5. visualize the main result

## What to add here next

To make this page truly useful, copy the essential commands from your core notebook and keep this page under 5 minutes of reading time.

Recommended additions:

- exact notebook link
- one toy example dataset
- expected runtime and GPU requirement
- one output figure
- one troubleshooting note for import/version issues

## Core demo

The repository already points users to a **STEER Core Pipeline Demo** notebook.[^tutorials]
That notebook should become the source for this page.

[^tutorials]: The README states that STEER provides an end-to-end core pipeline notebook and platform-specific preprocessing tutorials. citeturn259328view1
