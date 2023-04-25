import jax
import jax.numpy as jnp
import numpy as np


def ols_bcd(X, y, tol=1e-8, max_iters=10000):
    p = X.shape[-1]
    beta = np.zeros(p)
    X_norm_sq = np.sum(X ** 2, axis=0)
    delta = np.inf
    i = 0
    resid = np.copy(y)
    
    while (i < max_iters) and (delta > tol):
        delta = 0
        for j in range(p):
            bj_new = (X[:, j] @ resid) / X_norm_sq[j] + beta[j]
            delta += (bj_new - beta[j]) ** 2 * X_norm_sq[j]
            resid -= X[:, j] * (bj_new - beta[j])
            beta[j] = bj_new
        delta /= p
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
        alphas = curr_update ** 2 * X_norm_sq
        #alphas = alphas == np.max(alphas)
        alphas = alphas / jnp.sum(alphas)
        beta_new = beta + alphas * curr_update
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


def ols_epbcd(X, y, n_batches=16, tol=1e-8, max_iters=10000):
    p = X.shape[-1]
    X_norm_sq = jnp.sum(X ** 2, axis=0)
    batch_size = p // n_batches
    remainder = p - n_batches * batch_size

    i = 0
    beta = np.zeros(p) 
    resid = np.copy(y)
    deltas = np.zeros(n_batches)
    delta = jnp.inf
    alphas = np.zeros(n_batches)
    
    while (i < max_iters) and (delta > tol):
        curr_beta = np.copy(beta)

        # batch process each chunk of blocks
        for b in range(n_batches):
            begin = b * batch_size
            end = begin + batch_size + (b == n_batches-1) * remainder

            # coordinate descent on the batch
            curr_resid = np.copy(resid)
            for j in range(begin, end):
                bj_new = (X[:, j] @ curr_resid) / X_norm_sq[j] + curr_beta[j]
                beta_diff = bj_new - curr_beta[j]
                deltas[b] += beta_diff ** 2 * X_norm_sq[j]
                curr_resid -= X[:, j] * beta_diff
                curr_beta[j] = bj_new

            deltas[b] /= end - begin
    
        for b in range(n_batches):
            begin = b * batch_size
            end = begin + batch_size + (b == n_batches-1) * remainder
            #alphas = curr_update ** 2 * X_norm_sq
            #alphas /= np.sum(alphas)
            #alpha = 1./n_batches
            alphas[b] = np.max((curr_beta[begin:end] - beta[begin:end])** 2 * X_norm_sq[begin:end])

        alphas = alphas == np.max(alphas)
        #alphas = alphas / np.sum(alphas)

        # combine chunks
        for b in range(n_batches):
            begin = b * batch_size
            end = begin + batch_size + (b == n_batches-1) * remainder
            curr_beta[begin:end] = beta[begin:end] + alphas[b] * (curr_beta[begin:end] - beta[begin:end])

        resid -= X @ (curr_beta - beta)
        beta = curr_beta
        delta = np.mean(deltas)
        deltas[...] = 0
        i += 1
        print(delta)

    return beta, resid, delta, i