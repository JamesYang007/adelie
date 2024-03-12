<h1 align="center">
<img src="https://raw.githubusercontent.com/JamesYang007/adelie/main/docs/logos/adelie-penguin.svg" width="500">
</h1><br>

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/JamesYang007/adelie/test_docs.yml)
[![PyPI Downloads](https://img.shields.io/pypi/dm/adelie.svg?label=PyPI%20downloads)](https://pypi.org/project/adelie/)
![versions](https://img.shields.io/pypi/pyversions/adelie.svg)
![PyPI - Version](https://img.shields.io/pypi/v/adelie)
![GitHub Release](https://img.shields.io/github/v/release/JamesYang007/adelie)

Adelie is a fast and flexible Python package for solving group elastic net problems. 

- **Installation**: [https://jamesyang007.github.io/adelie/notebooks/installation.html](https://jamesyang007.github.io/adelie/notebooks/installation.html)
- **Documentation**: [https://jamesyang007.github.io/adelie](https://jamesyang007.github.io/adelie/)
- **Source code**: [https://github.com/JamesYang007/adelie](https://github.com/JamesYang007/adelie)
- **Issue Tracker**: [https://github.com/JamesYang007/adelie/issues](https://github.com/JamesYang007/adelie/issues)

It offers a general purpose group elastic net solver, 
a wide range of matrix classes that can exploit special structure to allow large-scale inputs,
and an assortment of generalized linear model (GLM) classes for fitting various types of data.
These matrix and GLM classes can be extended by the user for added flexibility.
Many inner routines such as matrix-vector products
and gradient, hessian, and loss of GLM functions have been heavily optimized and parallelized.
Algorithmic optimizations such as the pivot rule for screening variables
and the proximal Newton method have been carefully tuned for convergence and numerical stability.