
#' Group basil function
#' @export
group_basil <- function(
    X,
    y,
    groups,
    group_sizes,
    alpha=1,
    penalty=sqrt(group_sizes),
    user_lmdas=double(0),
    max_n_lambdas=100,
    n_lambdas_iter=5,
    use_strong_rule=TRUE,
    do_early_exit=TRUE,
    verbose_diagnostic=FALSE,
    delta_strong_size=5,
    max_strong_size=ncol(X),
    max_n_cds=as.integer(1e5),
    thr=1e-7,
    cond_0_thresh=1e-3,
    cond_1_thresh=1e-3,
    newton_tol=1e-8,
    newton_max_iters=100,
    min_ratio=1e-2,
    n_threads=16
)
{
    group_basil_naive__(
        X, y, as.integer(groups), as.integer(group_sizes), alpha, penalty,
        user_lmdas, max_n_lambdas, n_lambdas_iter,
        use_strong_rule, do_early_exit, verbose_diagnostic,
        delta_strong_size, max_strong_size,
        max_n_cds, thr, cond_0_thresh, cond_1_thresh,
        newton_tol, newton_max_iters, min_ratio, n_threads
    )
}