import jax
import jax.numpy as jnp
import numpy as np


def ols_bcd(X, y, beta0, tol=1e-8, max_iters=10000):
    p = X.shape[-1]
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0
    X_norm_sq = np.sum(X ** 2, axis=0)
    delta = np.inf
    i = 0
    resid = y - X @ beta
    
    while (i < max_iters) and (delta > tol):
        delta = 0
        for j in range(p):
            bj_new = (X[:, j] @ resid) / X_norm_sq[j] + beta[j]
            delta += (bj_new - beta[j]) ** 2 * X_norm_sq[j]
            resid -= X[:, j] * (bj_new - beta[j])
            beta[j] = bj_new
        delta /= p
        print(delta)
        i += 1
        
    return beta, resid, delta, i


def ols_pbcd(X, y, tol=1e-8, max_iters=10000):
    beta = jnp.zeros(X.shape[-1])
    X_norm_sq = jnp.sum(X ** 2, axis=0)
    #alphas = jnp.full(X.shape[-1], 1/X.shape[-1])
    
    def cond_fun(args):
        _, _, delta, i = args
        return (delta > tol) & (i < max_iters)
    
    def body_fun(args):
        beta, r, _, i = args
        curr_update = (X.T @ r) / X_norm_sq

        b = curr_update * X_norm_sq
        A_alphas = jnp.linalg.solve(X.T @ X, b)

        beta_new = beta + A_alphas
        r_new = y - X @ beta_new
        
        delta = jnp.max((r_new - r) ** 2)
        return beta_new, r_new, delta, i+1
        
    init_val = (beta, y, jnp.inf, 0)
    
    beta, resid, delta, i = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=init_val,
    )

    return beta, resid, delta, i


def ols_epbcd(X, y, n_batches=16, tol=1e-8, max_iters=10000, method='greedy', epochs=1):
    n, p = X.shape
    X_norm_sq = jnp.sum(X ** 2, axis=0)
    y_norm_sq = np.linalg.norm(y) ** 2
    batch_size = p // n_batches
    remainder = p - n_batches * batch_size

    i = 0
    beta = np.zeros(p) 
    resid = np.copy(y)
    #deltas = np.zeros(n_batches)
    delta = jnp.inf
    alphas = np.zeros(n_batches)
    resid_diffs = np.zeros((n_batches, n))
    
    while (i < max_iters) and (delta > tol):
        curr_beta = np.copy(beta)

        # batch process each chunk of blocks
        for b in range(n_batches):
            begin = b * batch_size
            end = begin + batch_size + (b == n_batches-1) * remainder

            # coordinate descent on the batch
            resid_diffs[b] = resid
            for _ in range(epochs):
                for j in range(begin, end):
                    bj_new = (X[:, j] @ resid_diffs[b]) / X_norm_sq[j] + curr_beta[j]
                    beta_diff = bj_new - curr_beta[j]
                    #deltas[b] += beta_diff ** 2 * X_norm_sq[j]
                    resid_diffs[b] -= X[:, j] * beta_diff
                    curr_beta[j] = bj_new
            resid_diffs[b] = resid - resid_diffs[b]

            #deltas[b] /= end - begin
    
        if method == 'greedy':
            i_star = np.argmin(np.linalg.norm(resid[None] - resid_diffs, axis=-1))
            alphas = np.zeros(n_batches)
            alphas[i_star] = 1
        else:
            alphas = np.linalg.solve(resid_diffs @ resid_diffs.T, resid_diffs @ resid)

        #eta = np.copy(resid)
        #y_hat_norm_sq = np.linalg.norm(y_hats, axis=-1) ** 2
        #alphas[...] = 0
        #delta = np.inf
        #while (i < max_iters) and (delta > tol):
        #    delta = 0
        #    for b in range(n_batches):
        #        begin = b * batch_size
        #        end = begin + batch_size + (b == n_batches-1) * remainder
        #        #alphas = curr_update ** 2 * X_norm_sq
        #        #alphas /= np.sum(alphas)
        #        #alpha = 1./n_batches
        #        #alphas[b] = np.max((curr_beta[begin:end] - beta[begin:end])** 2 * X_norm_sq[begin:end])
        #        alpha_new = alphas[b] + y_hats[b] @ eta / y_hat_norm_sq[b]
        #        dalpha = alpha_new - alphas[b]
        #        delta = max(delta, dalpha ** 2 * y_hat_norm_sq[b])
        #        eta -= y_hats[b] * dalpha
        #        alphas[b] = alpha_new
        #    i += 1

        # combine chunks
        for b in range(n_batches):
            begin = b * batch_size
            end = begin + batch_size + (b == n_batches-1) * remainder
            curr_beta[begin:end] = beta[begin:end] + alphas[b] * (curr_beta[begin:end] - beta[begin:end])

        curr_fit = np.linalg.norm(resid) ** 2
        resid -= X @ (curr_beta - beta)
        delta = (curr_fit - np.linalg.norm(resid) ** 2) / y_norm_sq
        beta = curr_beta
        #deltas[...] = 0
        i += 1
        print(delta)

    return beta, resid, delta, i


def ols_combined(X, y, n_batches=16, tol=1e-5, max_iters=10000, method='opt', epochs=1):
    beta, resid, delta, i = ols_epbcd(X, y, n_batches, tol, max_iters, method, epochs)
    return ols_bcd(X, y, beta, tol=1e-9, max_iters=max_iters)