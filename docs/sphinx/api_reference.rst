API Reference
=============


adelie.bcd
----------


.. currentmodule:: adelie.bcd


.. autosummary::
    :toctree: generated/


    objective
    root
    root_lower_bound
    root_upper_bound
    root_function
    solve


adelie.data
-----------------


.. currentmodule:: adelie.data


.. autosummary::
    :toctree: generated/


    create_dense
    create_snp_unphased
    create_snp_phased_ancestry


adelie.diagnostic
-----------------


.. currentmodule:: adelie.diagnostic


.. autosummary::
    :toctree: generated/


    residuals
    gradients
    gradient_norms
    gradient_scores
    coefficient
    plot_coefficients
    plot_devs
    plot_set_sizes
    plot_benchmark
    plot_kkt
    Diagnostic


adelie.io
---------


.. currentmodule:: adelie.io


.. autosummary::
    :toctree: generated/


    snp_unphased
    snp_phased_ancestry


adelie.matrix
-------------


.. currentmodule:: adelie.matrix


.. autosummary::
    :toctree: generated/


    dense
    concatenate
    cov_lazy
    snp_unphased
    snp_phased_ancestry


adelie.state
------------


.. currentmodule:: adelie.state


.. autosummary::
    :toctree: generated/


    deduce_states
    gaussian_pin_cov
    gaussian_pin_naive
    gaussian_naive
    glm_naive


adelie.solver
-------------


.. currentmodule:: adelie.solver


.. autosummary::
    :toctree: generated/


    objective
    solve_gaussian_pin
    solve_gaussian
    grpnet


Internal
--------


.. currentmodule:: adelie


.. autosummary::
    :toctree: generated/
    

    adelie_core.state.StateGaussianNaive64
    adelie_core.state.StateGaussianPinBase64
    adelie_core.state.StateGaussianPinCov64
    adelie_core.state.StateGaussianPinNaive64
    matrix.MatrixCovBase64
    matrix.MatrixNaiveBase64
    state.base