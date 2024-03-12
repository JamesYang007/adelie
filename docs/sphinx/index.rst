Adelie documentation
====================

.. toctree::
   :maxdepth: 1
   :hidden:

   api_reference
   user_guide


**Version**: |release|

**Useful links**:
`Installation <https://jamesyang007.github.io/adelie/notebooks/installation.html>`_ |
`Source Repository <https://github.com/JamesYang007/adelie>`_ |
`Issue Tracker <https://github.com/JamesYang007/adelie/issues>`_ 


Adelie is a fast and flexible Python package for solving group elastic net problems. 
It offers a general purpose group elastic net solver, 
a wide range of matrix classes that can exploit special structure to allow large-scale inputs,
and an assortment of generalized linear model (GLM) classes for fitting various types of data.
These matrix and GLM classes can be extended by the user for added flexibility.
Many inner routines such as matrix-vector products
and gradient, hessian, and loss of GLM functions have been heavily optimized and parallelized.
Algorithmic optimizations such as the pivot rule for screening variables
and the proximal Newton method have been carefully tuned for convergence and numerical stability.


.. grid:: 2

    .. grid-item-card::
        :img-top: ./_static/index-images/user_guide.svg

        User guide
        ^^^^^^^^^^

        The user guide provides an introduction to the relevant mathematical background
        as well as an in-depth guide on the usage of Adelie.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: ./_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The API reference contains a comprehensive list of functions and classes included in Adelie
        with detailed descriptions of how they work.

        +++

        .. button-ref:: api_reference
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide
