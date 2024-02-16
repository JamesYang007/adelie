API Reference
=============


adelie.bcd
----------


.. currentmodule:: adelie.bcd


.. autosummary::
    :toctree: generated/


    objective
    root
    root_function
    root_lower_bound
    root_upper_bound
    solve


adelie.data
-----------------


.. currentmodule:: adelie.data


.. autosummary::
    :toctree: generated/


    dense
    snp_phased_ancestry
    snp_unphased


adelie.diagnostic
-----------------


.. currentmodule:: adelie.diagnostic


.. autosummary::
    :toctree: generated/


    coefficient
    diagnostic
    gradients
    gradient_norms
    gradient_scores
    objective
    plot_benchmark
    plot_coefficients
    plot_devs
    plot_kkt
    plot_set_sizes
    predict
    residuals


adelie.glm
----------


.. currentmodule:: adelie.glm


.. autosummary::
    :toctree: generated/


    binomial
    cox
    gaussian
    multigaussian
    multinomial
    poisson


adelie.io
---------


.. currentmodule:: adelie.io


.. autosummary::
    :toctree: generated/


    snp_phased_ancestry
    snp_unphased


adelie.matrix
-------------


.. currentmodule:: adelie.matrix


.. autosummary::
    :toctree: generated/


    concatenate
    cov_lazy
    dense
    kronecker_eye
    snp_phased_ancestry
    snp_unphased


adelie.state
------------


.. currentmodule:: adelie.state


.. autosummary::
    :toctree: generated/


    gaussian_naive
    gaussian_pin_cov
    gaussian_pin_naive
    glm_naive
    multigaussian_naive
    multiglm_naive


adelie.solver
-------------


.. currentmodule:: adelie.solver


.. autosummary::
    :toctree: generated/


    grpnet


Internal
--------


.. currentmodule:: adelie


.. autosummary::
    :toctree: generated/
    

    adelie_core.state.StateGaussianPinBase64
    adelie_core.state.StateGaussianPinCov64
    adelie_core.state.StateGaussianPinNaive64
    adelie_core.state.StateGaussianNaive64
    adelie_core.state.StateGlmNaive64
    adelie_core.state.StateMultiGaussianNaive64
    adelie_core.state.StateMultiGlmNaive64
    glm.GlmBase64
    glm.GlmMultiBase64
    matrix.MatrixCovBase64
    matrix.MatrixNaiveBase64
    state.base