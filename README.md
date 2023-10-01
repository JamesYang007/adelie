# Adelie

This repository contains research about group-lasso.
Specifically, we hope to consolidate all empirical studies about how group-lasso
regarding its performance based on different methods, literature overview, and application results
to decide which implementation/application makes most sense depending on the situation.

## Installation

MacOS users must have `OpenMP` installed in their system as a prerequisite.
The simplest method is to install with `brew`:
```
brew install libomp
```

Install the stable version with `pip`:
```
pip install adelie
```

For the bleeding edge version, follow the instructions in [Developer Installation](#developer-installation).

## Developer Installation

1. Install conda. We recommend installing `Mambaforge`, which is a conda installation with
`mamba` installed by default and set to use `conda-forge` as the default set of package repositories.

2. Clone the git repo:
    ```
    git clone git@github.com:JamesYang007/adelie.git
    ```

3. Set up `adelie` conda environment. The list of packages that will be installed in the environment
is in `pyproject.toml`.
    ```
    mamba update -y conda mamba
    mamba env create
    conda activate adelie
    poetry config virtualenvs.create false --local
    poetry install --no-root
    ```

4. Install `adelie` in editable mode:
    ```
    pip install -e .
    ```

## References

- [Strong Rules for Discarding Predictors in Lasso-type Problems](https://www.stat.cmu.edu/~ryantibs/papers/strongrules.pdf)
- [sparsegl: An R Package for Estimating Sparse Group Lasso](https://arxiv.org/abs/2208.02942)
- [Hybrid safe-strong rules for efficient optimization in lasso-type problems](https://arxiv.org/abs/1704.08742)
- [A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems](https://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf)
- [STANDARDIZATION AND THE GROUP LASSO PENALTY](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4527185/)
    - `Many works explicitly do not sphere (Puig, Wiesel, and Hero (2009), Foygel and Drton (2010), Jacob, Obozinski, and Vert (2009), Hastie, Tibshirani, and Friedman (2008), among others), and many more make no mention of normalization`.
- Successive over-relaxation (SOR)!