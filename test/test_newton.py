import grpglmnet as gl
import numpy as np

def test_newton_abs():
    L = np.array([2.66325906971018])
    v = np.array([-3.271350652744633])
    l1 = 2.980732876975005
    l2 = 0.0
    tol = 1e-16
    max_iters = 100
    out = gl.newton_abs_solver(
        L, v, l1, l2, tol, max_iters
    )
    actual = out['beta']
    expected = [-0.1091211061946196]
    print(actual)
    print(expected)
    print(out['iters'])
    assert np.allclose(actual, expected)
