from scipy.special import xlogy
from scipy.stats import norm
import adelie.glm as glm
import adelie.configs as configs
import adelie.adelie_core as core
import numpy as np


def run_common_test(
    model,
    model_exp,
):
    shape = model.y.shape

    eta = np.random.normal(0, 1, shape)

    # test grad
    grad = np.empty(shape)
    grad_exp = np.empty(shape)
    model.gradient(eta, grad)
    model_exp.gradient(eta, grad_exp)
    assert np.allclose(grad, grad_exp)

    # test hessian
    hess = np.empty(shape)
    hess_exp = np.empty(shape)
    model.hessian(eta, grad, hess)
    model_exp.hessian(eta, grad, hess_exp)
    assert np.allclose(hess, hess_exp)

    # test inv_hessian_gradient
    inv_hess_grad = np.empty(shape)
    inv_hess_grad_exp = np.empty(shape)
    model.inv_hessian_gradient(eta, grad, hess, inv_hess_grad)
    model_exp.inv_hessian_gradient(eta, grad, hess, inv_hess_grad_exp)
    assert np.allclose(inv_hess_grad, inv_hess_grad_exp, atol=1e-6)

    # test loss
    loss = model.loss(eta)
    loss_exp = model_exp.loss(eta)
    assert np.allclose(loss, loss_exp)

    # test loss_full
    loss_full = model.loss_full()
    loss_full_exp = model_exp.loss_full()
    assert np.allclose(loss_full, loss_full_exp)


def run_subset_test(
    model,
    model_exp,
    subset,
):
    shape = model.y.shape
    shape_exp = model_exp.y.shape

    eta = np.random.normal(0, 1, shape)
    eta_sub = eta[subset]

    # test grad
    grad = np.empty(shape)
    grad_exp = np.empty(shape_exp)
    model.gradient(eta, grad)
    model_exp.gradient(eta_sub, grad_exp)
    assert np.allclose(grad[subset], grad_exp)

    # test hessian
    hess = np.empty(shape)
    hess_exp = np.empty(shape_exp)
    model.hessian(eta, grad, hess)
    model_exp.hessian(eta_sub, grad_exp, hess_exp)
    assert np.allclose(hess[subset], hess_exp)

    # test inv_hessian_gradient
    inv_hess_grad = np.empty(shape)
    inv_hess_grad_exp = np.empty(shape_exp)
    model.inv_hessian_gradient(eta, grad, hess, inv_hess_grad)
    model_exp.inv_hessian_gradient(eta_sub, grad_exp, hess_exp, inv_hess_grad_exp)
    assert np.allclose(inv_hess_grad[subset], inv_hess_grad_exp, atol=1e-6)

    # test loss
    loss = model.loss(eta)
    loss_exp = model_exp.loss(eta_sub)
    assert np.allclose(loss, loss_exp)

    # test loss_full
    loss_full = model.loss_full()
    loss_full_exp = model_exp.loss_full()
    assert np.allclose(loss_full, loss_full_exp)


class GlmTest():
    def inv_hessian_gradient(self, eta, grad, hess, inv_hess_grad):
        inv_hess_grad[...] = grad / (np.maximum(hess, 0) + configs.Configs.hessian_min * (hess <= 0))


# =====================================================================================
# TEST gaussian
# =====================================================================================


class GlmTestGaussian(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        grad[...] = self.weights * (self.y - eta)
    
    def hessian(self, eta, grad, hess):
        hess[...] = self.weights

    def loss(self, eta):
        return np.sum(self.weights * (-self.y * eta + eta ** 2 / 2))

    def loss_full(self):
        return -np.sum(self.weights * self.y ** 2 / 2)


def test_gaussian():
    def _test(n, seed=0):
        np.random.seed(seed)
        y = np.random.normal(0, 1, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.gaussian(y=y, weights=w)
        model_exp = GlmTestGaussian(y=y, weights=w)
        run_common_test(model, model_exp)

        subset = w != 0
        y, w = y[subset], w[subset]
        model_exp = glm.gaussian(y=y, weights=w)
        run_subset_test(model, model_exp, subset)

    ns = [1, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)


# =====================================================================================
# TEST binomial
# =====================================================================================


class GlmTestBinomialLogit(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        mu = 1 / (1 + np.exp(-eta))
        grad[...] = self.weights * (self.y - mu)
    
    def hessian(self, eta, grad, hess):
        mu = 1 / (1 + np.exp(-eta))
        hess[...] = self.weights * mu * (1 - mu)

    def loss(self, eta):
        A = np.log(1 + np.exp(eta))
        return np.sum(self.weights * (-self.y * eta + A))

    def loss_full(self):
        return -np.sum(
            self.weights * (xlogy(self.y, self.y) + xlogy(1-self.y, 1-self.y))
        )


class GlmTestBinomialProbit(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        Phis = norm.cdf(eta)
        grad[...] = self.weights * (
            self.y / Phis - (1-self.y) / (1-Phis)
        ) * norm.pdf(eta)
    
    def hessian(self, eta, grad, hess):
        y = self.y
        Phis = norm.cdf(eta)
        phis = norm.pdf(eta)
        hess[...] = self.weights * (
            (y / Phis ** 2 + (1-y) / (1-Phis) ** 2) * phis ** 2
            +
            (y / Phis - (1-y) / (1-Phis)) * phis * eta
        )

    def loss(self, eta):
        Phis = norm.cdf(eta)
        return -np.sum(self.weights * (
            self.y * np.log(Phis) + (1-self.y) * np.log(1-Phis)
        ))

    def loss_full(self):
        return -np.sum(
            self.weights * (xlogy(self.y, self.y) + xlogy(1-self.y, 1-self.y))
        )


def test_binomial():
    def _test(n, link, binary, seed=0):
        np.random.seed(seed)
        if binary:
            y = np.random.binomial(1, 0.5, n)
        else:
            y = np.random.uniform(0, 1, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.binomial(y=y, weights=w, link=link, dtype=np.float64)
        test_glm = {
            "logit": GlmTestBinomialLogit,
            "probit": GlmTestBinomialProbit,
        }[link]
        model_exp = test_glm(y=y, weights=w)
        run_common_test(model, model_exp)

        subset = w != 0
        y, w = y[subset], w[subset]
        model_exp = glm.binomial(y=y, weights=w, link=link, dtype=np.float64)
        run_subset_test(model, model_exp, subset)

    links = ["logit", "probit"]
    ns = [1, 2, 5, 10, 20, 100]
    for link in links:
        for n in ns:
            _test(n, link, True)
            _test(n, link, False)


# =====================================================================================
# TEST poisson
# =====================================================================================


class GlmTestPoisson(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        mu = np.exp(eta)
        grad[...] = self.weights * (self.y - mu)
    
    def hessian(self, eta, grad, hess):
        hess[...] = self.weights * np.exp(eta)

    def loss(self, eta):
        A = np.exp(eta)
        return np.sum(self.weights * (-self.y * eta + A))

    def loss_full(self):
        return np.sum(self.weights * (-xlogy(self.y, self.y) + self.y))


def test_poisson():
    def _test(n, seed=0):
        np.random.seed(seed)
        y = np.random.poisson(1, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.poisson(y=y, weights=w, dtype=np.float64)
        model_exp = GlmTestPoisson(y=y, weights=w)
        run_common_test(model, model_exp)

        subset = w != 0
        y, w = y[subset], w[subset]
        model_exp = glm.poisson(y=y, weights=w, dtype=np.float64)
        run_subset_test(model, model_exp, subset)

    ns = [1, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)


# =====================================================================================
# TEST cox
# =====================================================================================


def test_cox_partial_sum_fwd():
    def _test(n, m, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        v = np.random.uniform(0, 1, n)
        s = np.sort(np.random.uniform(0, 1, n))
        t = np.sort(np.random.uniform(0, 1, m))

        out = np.empty(m+1)
        core.glm.GlmCox64._partial_sum_fwd(v, s, t, out)
        expected = np.sum((s[None] <= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[1:], expected)

        # discrete time (create ties)
        s = np.sort(np.random.choice(20, n))
        t = np.sort(np.random.choice(20, m))
        out = np.empty(m+1)
        core.glm.GlmCox64._partial_sum_fwd(v, s, t, out)
        expected = np.sum((s[None] <= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[1:], expected)

    ns = [0, 2, 5, 10, 20, 100]
    ms = [0, 4, 5, 30, 50, 150]
    for n in ns:
        for m in ms:
            _test(n, m)


def test_cox_partial_sum_bwd():
    def _test(n, m, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        v = np.random.uniform(0, 1, n)
        s = np.sort(np.random.uniform(0, 1, n))
        t = np.sort(np.random.uniform(0, 1, m))

        out = np.empty(m+1)
        core.glm.GlmCox64._partial_sum_bwd(v, s, t, out)
        expected = np.sum((s[None] >= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[:-1], expected)

        # discrete time (create ties)
        s = np.sort(np.random.choice(20, n))
        t = np.sort(np.random.choice(20, m))
        out = np.empty(m+1)
        core.glm.GlmCox64._partial_sum_bwd(v, s, t, out)
        expected = np.sum((s[None] >= t[:, None]) * v[None], axis=-1)
        assert np.allclose(out[:-1], expected)

    ns = [0, 2, 5, 10, 20, 100]
    ms = [0, 4, 5, 30, 50, 150]
    for n in ns:
        for m in ms:
            _test(n, m)


def test_cox_at_risk_sum():
    def _test(n, m, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        a = np.random.uniform(0, 1, n)
        s = np.random.uniform(0, 1, n)
        t = np.random.uniform(s, 1, n)
        u = np.random.uniform(0, 1, m)
        s_sorted = np.sort(s)
        t_sorted = np.sort(t)
        u_sorted = np.sort(u)
        a_s = a[np.argsort(s)]
        a_t = a[np.argsort(t)]
        out = np.empty(m)
        out1 = np.empty(m+1)
        out2 = np.empty(m+1)

        core.glm.GlmCox64._at_risk_sum(
            a_s, a_t, s_sorted, t_sorted, u_sorted, out, out1, out2,
        )
        expected = np.sum(
            (s[None] < u[:, None]) * (u[:, None] <= t[None]) * a[None], 
            axis=-1,
        )[np.argsort(u)]
        assert np.allclose(out, expected)

        # discrete time (create ties)
        s = np.random.choice(5, n)
        t = 1 + s + np.random.choice(5, n)
        u = np.random.choice(10, m)
        s_sorted = np.sort(s)
        t_sorted = np.sort(t)
        u_sorted = np.sort(u)
        a_s = a[np.argsort(s)]
        a_t = a[np.argsort(t)]
        core.glm.GlmCox64._at_risk_sum(
            a_s, a_t, s_sorted, t_sorted, u_sorted, out, out1, out2,
        )
        expected = np.sum(
            (s[None] < u[:, None]) * (u[:, None] <= t[None]) * a[None], 
            axis=-1,
        )[np.argsort(u)]
        assert np.allclose(out, expected)

    ns = [0, 2, 5, 10, 20, 100]
    ms = [0, 4, 5, 30, 50, 150]
    for n in ns:
        for m in ms:
            _test(n, m)


def test_cox_nnz_event_ties_sum():
    def _test(n, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        t = np.sort(np.random.uniform(0, 1, n))
        status = np.random.binomial(1, 0.5, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        a = np.random.normal(0, 1, n)

        out = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(a, t, status, w, out)
        expected = (
            np.sum((t[None] == t[:, None]) * ((w != 0) * status * a)[None], axis=-1)
        )
        expected[(status == 0) | (w == 0)] = 0
        assert np.allclose(out, expected)

        # discrete time (create ties)
        t = np.sort(np.random.choice(20, n))
        out = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(a, t, status, w, out)
        expected = (
            np.sum((t[None] == t[:, None]) * ((w != 0) * status * a)[None], axis=-1)
        )
        expected[(status == 0) | (w == 0)] = 0
        assert np.allclose(out, expected)

    ns = [0, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)


def test_cox_scale():
    def _test(n, seed=0):
        np.random.seed(seed)

        # continuous time (unique times a.s.)
        t = np.sort(np.random.uniform(0, 1, n))
        status = np.random.binomial(1, 0.5, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0

        out = np.empty(n)
        core.glm.GlmCox64._scale(t, status, w, "efron", out)
        ta = t[(status != 0) & (w != 0)]
        _,  unique_counts = np.unique(ta, return_counts=True)
        expected = np.zeros(n)
        expected[(status != 0) & (w != 0)] = np.concatenate([
            np.arange(c, dtype=float) / c
            for c in unique_counts
        ])
        assert np.allclose(out, expected)

        # discrete time (create ties)
        t = np.sort(np.random.choice(20, n))
        out = np.empty(n)
        core.glm.GlmCox64._scale(t, status, w, "efron", out)
        ta = t[(status != 0) & (w != 0)]
        _, unique_counts = np.unique(ta, return_counts=True)
        expected = np.zeros(n)
        expected[(status != 0) & (w != 0)] = np.concatenate([
            np.arange(c, dtype=float) / c
            for c in unique_counts
        ])
        assert np.allclose(out, expected)

    ns = [1, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)


class GlmTestCox(GlmTest):
    def __init__(
        self,
        start,
        stop,
        status,
        weights,
        tie_method="efron",
    ):
        self.start = start
        self.stop = stop
        self.status = status
        self.weights = weights
        self.tie_method = tie_method

        n = self.start.shape[0]

        self.stop_order = np.argsort(stop)
        self.inv_stop_order = np.argsort(self.stop_order)

        self.scale = np.empty(n)
        core.glm.GlmCox64._scale(
            stop[self.stop_order], 
            status[self.stop_order], 
            weights[self.stop_order], 
            tie_method, 
            self.scale,
        )
        self.scale = self.scale[self.inv_stop_order]

        self.weights_sum = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(
            self.weights[self.stop_order], 
            stop[self.stop_order], 
            status[self.stop_order], 
            weights[self.stop_order], 
            self.weights_sum,
        )
        self.weights_sum = self.weights_sum[self.inv_stop_order]

        self.weights_size = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(
            np.ones(n),
            stop[self.stop_order], 
            status[self.stop_order], 
            weights[self.stop_order], 
            self.weights_size,
        )
        self.weights_size = self.weights_size[self.inv_stop_order]

        self.weights_mean = np.divide(
            self.weights_sum,
            self.weights_size,
            where=self.weights_size > 0,
        )
        self.weights_mean[self.weights_size == 0] = 0


    def gradient(self, eta, grad):
        s = self.start
        t = self.stop
        d = self.status
        w = self.weights
        sigma = self.scale
        w_mean = self.weights_mean

        z = w * np.exp(eta)
        risk_sums = (
            np.sum(((s[None] < t[:, None]) & (t[:, None] <= t[None])) * z[None], axis=-1) 
            -
            sigma * np.sum((t[:, None] == t[None]) * (d * z)[None], axis=-1)
        )
        grad_scale = np.sum(
            ((s[:, None] < t[None]) * (t[None] <= t[:, None]) - sigma[None] * (t[None] == t[:, None]) * d[:, None]) *
            np.divide(
                d * w_mean,
                risk_sums,
                where=risk_sums > 0,
            )[None],
            axis=-1
        )
        grad[...] = d * w -  grad_scale * z

    def hessian(self, eta, grad, hess):
        n = eta.shape[0]
        s = self.start
        t = self.stop
        d = self.status
        w = self.weights
        sigma = self.scale
        weights_mean = self.weights_mean

        grad = np.empty(n)
        self.gradient(eta, grad)

        z = w * np.exp(eta)
        risk_sums = (
            np.sum(((s[None] < t[:, None]) & (t[:, None] <= t[None])) * z[None], axis=-1) 
            -
            sigma * np.sum((t[:, None] == t[None]) * (d * z)[None], axis=-1)
        )
        hess_scale = np.sum(
            d * weights_mean *
            np.divide(
                ((s[:, None] < t[None]) * (t[None] <= t[:, None]) - sigma[None] * (t[None] == t[:, None]) * d[:, None]),
                risk_sums,
                where=risk_sums > 0
            ) ** 2,
            axis=-1,
        )
        hess[...] = d * w - grad - hess_scale * z ** 2

    def loss(self, eta):
        s = self.start
        t = self.stop
        d = self.status
        w = self.weights
        sigma = self.scale
        w_mean = self.weights_mean
        z = w * np.exp(eta)
        risk_sums = (
            np.sum(((s[None] < t[:, None]) & (t[:, None] <= t[None])) * z[None], axis=-1) 
            -
            sigma * np.sum((t[:, None] == t[None]) * (d * z)[None], axis=-1)
        )
        return (
            - np.sum(d * w * eta)
            + np.sum(d * w_mean * np.log(risk_sums + (risk_sums <= 0)))
        )

    def loss_full(self):
        d = self.status
        sigma = self.scale
        w_mean = self.weights_mean
        w_sum = self.weights_sum
        return np.sum(d * w_mean * np.log(w_sum * (1 - sigma) + (w_sum <= 0)))


def test_cox():
    def _test(n, seed=0):
        np.random.seed(seed)

        # discrete time (create ties)
        s = np.random.choice(20, n)
        t = 1 + s + np.random.choice(20, n)
        d = np.random.binomial(1, 0.5, n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.cox(
            start=s,
            stop=t,
            status=d,
            weights=w,
            dtype=np.float64,
        )
        model_exp = GlmTestCox(
            start=s,
            stop=t,
            status=d,
            weights=w,
        )
        run_common_test(model, model_exp)

        subset = w != 0
        s, t, d, w = s[subset], t[subset], d[subset], w[subset]
        model_exp = glm.cox(
            start=s, 
            stop=t,
            status=d,
            weights=w,
            dtype=np.float64,
        )
        run_subset_test(model, model_exp, subset)

    ns = [1, 2, 5, 10, 20, 100]
    for n in ns:
        _test(n)


# =====================================================================================
# TEST multigaussian
# =====================================================================================


class GlmTestMultiGaussian(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        K = self.y.shape[-1]
        grad[...] = self.weights[:, None] * (self.y - eta) / K
    
    def hessian(self, eta, grad, hess):
        K = self.y.shape[-1]
        hess[...] = self.weights[:, None] / K

    def loss(self, eta):
        K = self.y.shape[-1]
        return np.sum(self.weights[:, None] * (-self.y * eta + eta ** 2 / 2)) / K

    def loss_full(self):
        K = self.y.shape[-1]
        return -np.sum(self.weights[:, None] * self.y ** 2 / 2) / K


def test_multigaussian():
    def _test(n, K, seed=0):
        np.random.seed(seed)
        y = np.random.normal(0, 1, (n, K))
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.multigaussian(y=y, weights=w)
        model_exp = GlmTestMultiGaussian(y=y, weights=w)
        run_common_test(model, model_exp)

        subset = w != 0
        y, w = y[subset], w[subset]
        model_exp = glm.multigaussian(y=y, weights=w)
        run_subset_test(model, model_exp, subset)

    ns = [1, 2, 5, 10, 20, 100]
    Ks = [1, 2, 3, 4]
    for n in ns:
        for K in Ks:
            _test(n, K)


# =====================================================================================
# TEST multinomial
# =====================================================================================


class GlmTestMultinomial(GlmTest):
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights

    def gradient(self, eta, grad):
        K = self.y.shape[-1]
        mu = np.exp(eta)
        mu = mu / np.sum(mu, axis=-1)[:, None]
        grad[...] = self.weights[:, None] * (self.y - mu) / K
    
    def hessian(self, eta, grad, hess):
        K = self.y.shape[-1]
        mu = np.exp(eta)
        mu = mu / np.sum(mu, axis=-1)[:, None]
        hess[...] = 2 * self.weights[:, None] / K * mu * (1 - mu)

    def loss(self, eta):
        K = self.y.shape[-1]
        A = np.log(np.sum(np.exp(eta), axis=-1))
        return np.sum(self.weights * (-np.sum(self.y * eta, axis=-1) + A)) / K

    def loss_full(self):
        K = self.y.shape[-1]
        return -np.sum(self.weights * np.sum(xlogy(self.y, self.y), axis=-1)) / K


def test_multinomial():
    def _test(n, K, binary=True, seed=0):
        np.random.seed(seed)
        if binary:
            y = np.random.multinomial(1, np.full(K, 1/K), n)
        else:
            y = np.random.dirichlet(np.ones(K), n)
        w = np.random.uniform(0, 1, n)
        w[np.random.binomial(1, 0.2, n).astype(bool)] = 0
        w[0] = 1
        w /= np.sum(w)
        model = glm.multinomial(y=y, weights=w, dtype=np.float64)
        model_exp = GlmTestMultinomial(y=y, weights=w)
        run_common_test(model, model_exp)

        subset = w != 0
        y, w = y[subset], w[subset]
        model_exp = glm.multinomial(y=y, weights=w, dtype=np.float64)
        run_subset_test(model, model_exp, subset)

    ns = [1, 2, 5, 10, 20, 100]
    Ks = [2, 3, 4]
    for n in ns:
        for K in Ks:
            _test(n, K, True)
            _test(n, K, False)
