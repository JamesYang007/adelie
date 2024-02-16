import adelie.glm as glm
import numpy as np


# =====================================================================================
# TEST cox
# =====================================================================================


def test_cox_partial_sum():
    def _test(n, m, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        v = np.random.uniform(0, 1, n)
        s = np.sort(np.random.uniform(0, 1, n))
        t = np.sort(np.random.uniform(0, 1, m))

        out = np.empty(m+1)
        cox = glm.cox()
        cox._partial_sum(v, s, t, out)
        expected = np.sum((s[None] >= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[:-1], expected)

        # discrete time (create ties)
        s = np.sort(np.random.choice(20, n))
        t = np.sort(np.random.choice(20, m))
        out = np.empty(m+1)
        cox = glm.cox()
        cox._partial_sum(v, s, t, out)
        expected = np.sum((s[None] >= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[:-1], expected)

    ns = [0, 2, 5, 10, 20, 100]
    ms = [0, 4, 5, 30, 50, 150]
    for n in ns:
        for m in ms:
            _test(n, m)


def test_cox_average_ties():
    def _test(n, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        v = np.random.uniform(0, 1, n)
        t = np.sort(np.random.uniform(0, 1, n))

        out = np.empty(n)
        cox = glm.cox()
        cox._average_ties(v, t, out)
        expected = (
            np.sum((t[None] == t[:, None]) * v[None], axis=-1) /
            np.sum((t[None] == t[:, None]), axis=-1)
        )
        assert np.allclose(out, expected)

        # discrete time (create ties)
        t = np.sort(np.random.choice(20, n))
        out = np.empty(n)
        cox = glm.cox()
        cox._average_ties(v, t, out)
        expected = (
            np.sum((t[None] == t[:, None]) * v[None], axis=-1) /
            np.sum((t[None] == t[:, None]), axis=-1)
        )
        assert np.allclose(out, expected)

    ns = [0, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)