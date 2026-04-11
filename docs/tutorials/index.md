# Tutorials

The STEER tutorial system is organized to match the actual structure of the GitHub repository. The main tutorial resources currently fall into three groups:

1. **Core pipeline**
2. **Main figure notebooks**
3. **Raw-data preprocessing**

## 1. Core pipeline

The main entry point for using STEER is the core demo notebook:

- `tutorials/demo_tutorial.ipynb`

This is the recommended starting point for most users. It introduces the standard workflow for running STEER on prepared input data.

You can access it in this documentation site through:

- **Core Pipeline Notebook**

You can also browse the original file on GitHub:

- [demo_tutorial.ipynb](https://github.com/lzygenomics/STEER/tree/master/tutorials/demo_tutorial.ipynb)

## 2. Main figure notebooks

The repository contains a dedicated notebook collection for the main figure-related analyses:

- `tutorials/NoteBook_Main_Figures/Figure2.ipynb`
- `tutorials/NoteBook_Main_Figures/Figure3.ipynb`
- `tutorials/NoteBook_Main_Figures/Figure4.ipynb`
- `tutorials/NoteBook_Main_Figures/Figure5.ipynb`

These notebooks are especially useful for users who want to understand how the main analyses and figure-level results in the STEER study were generated.

Please see the **Main Figures** page for a guided overview, or open the notebook pages directly from the site navigation.

Original GitHub files:

- [Figure2.ipynb](https://github.com/lzygenomics/STEER/tree/master/tutorials/NoteBook_Main_Figures/Figure2.ipynb)
- [Figure3.ipynb](https://github.com/lzygenomics/STEER/tree/master/tutorials/NoteBook_Main_Figures/Figure3.ipynb)
- [Figure4.ipynb](https://github.com/lzygenomics/STEER/tree/master/tutorials/NoteBook_Main_Figures/Figure4.ipynb)
- [Figure5.ipynb](https://github.com/lzygenomics/STEER/tree/master/tutorials/NoteBook_Main_Figures/Figure5.ipynb)

## 3. Raw-data preprocessing

If your input data does not yet contain `spliced` and `unspliced` layers, please begin with the preprocessing resources under:

- `tutorials/raw_data_processing/`

These tutorials are designed for generating the matrices needed before running the core STEER workflow.

This section includes platform-specific routes such as:

- Slide-seq
- 10x Visium
- Stereo-seq

Please see the **Raw Data Preprocessing** page for a structured summary.

## Recommended learning path

Choose your path according to your goal.

### I want to run STEER as quickly as possible

Start with:

- **Core Pipeline Notebook**

### I want to reproduce or understand the paper’s main analyses

Continue with:

- **Main Figures**
- **Figure 2 Notebook**
- **Figure 3 Notebook**
- **Figure 4 Notebook**
- **Figure 5 Notebook**

### My data does not yet contain spliced/unspliced matrices

Start with:

- **Raw Data Preprocessing**

## Tutorial philosophy

These documentation pages are intended to mirror the real repository layout rather than abstracting away the notebook structure. This makes it easier for users to move between the documentation site and the GitHub repository without losing context.
