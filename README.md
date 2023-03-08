# Group-Lasso-Study

This repository contains research about group-lasso.
Specifically, we hope to consolidate all empirical studies about how group-lasso
regarding its performance based on different methods, literature overview, and application results
to decide which implementation/application makes most sense depending on the situation.

## Setup Instructions

1. Install conda. We recommend installing `Mambaforge`, which is a conda installation with
`mamba` installed by default and set to use `conda-forge` as the default set of package repositories.

2. Clone the git repo:
```
git clone git@github.com:JamesYang007/Group-Lasso-Study.git
```

3. Set up `glstudy` conda environment. The list of packages that will be installed in the environment
is in `pyproject.toml`.
```
mamba update -y conda mamba
# create a development virtual environment with useful tools
mamba env create
conda activate glstudy
# the following line makes poetry install into already activated
# environment.
poetry config virtualenvs.create false --local
# install the imprint package plus development tools
poetry install --no-root
```


## References

- [Strong Rules for Discarding Predictors in Lasso-type Problems](https://www.stat.cmu.edu/~ryantibs/papers/strongrules.pdf)
- [sparsegl: An R Package for Estimating Sparse Group Lasso](https://arxiv.org/abs/2208.02942)
- [Hybrid safe-strong rules for efficient optimization in lasso-type problems](https://arxiv.org/abs/1704.08742)
- [A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems](https://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf)