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


    create_test_data_basil


adelie.diagnostic
-----------------


.. currentmodule:: adelie.diagnostic


.. autosummary::
    :toctree: generated/


    plot_coefficient
    plot_rsq
    plot_set_size
    plot_benchmark
    plot_kkt


adelie.matrix
-------------


.. currentmodule:: adelie.matrix


.. autosummary::
    :toctree: generated/


    naive_dense
    cov_dense
    cov_lazy


adelie.state
------------


.. currentmodule:: adelie.state


.. autosummary::
    :toctree: generated/


    deduce_states
    pin_cov
    pin_naive
    basil_naive


adelie.solver
-------------


.. currentmodule:: adelie.solver


.. autosummary::
    :toctree: generated/


    objective
    solve_pin
    solve_basil
    grpnet


Internal
--------


.. currentmodule:: adelie


.. autosummary::
    :toctree: generated/
    

    matrix.base
    matrix.MatrixCovBase32
    matrix.MatrixCovBase64
    matrix.MatrixNaiveBase32
    matrix.MatrixNaiveBase64
    state.base
    state.pin_cov_32
    state.pin_cov_64
    state.pin_naive_32
    state.pin_naive_64
    state.basil_naive_32
    state.basil_naive_64
    adelie_core.state.StateBasilBase32
    adelie_core.state.StateBasilBase64
    adelie_core.state.StatePinBase32
    adelie_core.state.StatePinBase64
