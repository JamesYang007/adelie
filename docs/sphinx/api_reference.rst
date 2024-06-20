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


adelie.configs
--------------


.. currentmodule:: adelie.configs


.. autosummary::
    :toctree: generated/


    set_configs


adelie.constraint
-----------------


.. currentmodule:: adelie.constraint


.. autosummary::
    :toctree: generated/


    lower


adelie.cv
---------


.. currentmodule:: adelie.cv


.. autosummary::
    :toctree: generated/


    CVGrpnetResult
    cv_grpnet


adelie.data
-----------


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

    DiagnosticCov
    DiagnosticNaive
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


    block_diag
    concatenate
    dense
    eager_cov
    interaction
    kronecker_eye
    lazy_cov
    one_hot
    snp_phased_ancestry
    snp_unphased
    sparse
    standardize
    subset


adelie.solver
-------------


.. currentmodule:: adelie.solver


.. autosummary::
    :toctree: generated/


    gaussian_cov
    grpnet


adelie.state
------------


.. currentmodule:: adelie.state


.. autosummary::
    :toctree: generated/


    gaussian_cov
    gaussian_naive
    gaussian_pin_cov
    gaussian_pin_naive
    glm_naive
    multigaussian_naive
    multiglm_naive


Internal
--------


.. currentmodule:: adelie


.. autosummary::
    :toctree: generated/
    

    adelie_core.configs.Configs
    adelie_core.constraint.ConstraintBase64
    adelie_core.constraint.ConstraintLowerUpper64
    adelie_core.glm.GlmBase64
    adelie_core.glm.GlmBinomialLogit64
    adelie_core.glm.GlmBinomialProbit64
    adelie_core.glm.GlmCox64
    adelie_core.glm.GlmGaussian64
    adelie_core.glm.GlmMultiBase64
    adelie_core.glm.GlmMultiGaussian64
    adelie_core.glm.GlmMultinomial64
    adelie_core.glm.GlmPoisson64
    adelie_core.matrix.MatrixCovBase64
    adelie_core.matrix.MatrixCovBlockDiag64
    adelie_core.matrix.MatrixCovDense64F
    adelie_core.matrix.MatrixCovLazyCov64F
    adelie_core.matrix.MatrixCovSparse64F
    adelie_core.matrix.MatrixNaiveBase64
    adelie_core.matrix.MatrixNaiveCConcatenate64
    adelie_core.matrix.MatrixNaiveRConcatenate64
    adelie_core.matrix.MatrixNaiveDense64F
    adelie_core.matrix.MatrixNaiveInteractionDense64F
    adelie_core.matrix.MatrixNaiveKroneckerEye64
    adelie_core.matrix.MatrixNaiveKroneckerEyeDense64F
    adelie_core.matrix.MatrixNaiveOneHotDense64F
    adelie_core.matrix.MatrixNaiveSNPPhasedAncestry64
    adelie_core.matrix.MatrixNaiveSNPUnphased64
    adelie_core.matrix.MatrixNaiveSparse64F
    adelie_core.matrix.MatrixNaiveStandardize64
    adelie_core.matrix.MatrixNaiveCSubset64
    adelie_core.matrix.MatrixNaiveRSubset64
    adelie_core.state.StateGaussianPinCov64
    adelie_core.state.StateGaussianPinNaive64
    adelie_core.state.StateGaussianCov64
    adelie_core.state.StateGaussianNaive64
    adelie_core.state.StateGlmNaive64
    adelie_core.state.StateMultiGaussianNaive64
    adelie_core.state.StateMultiGlmNaive64