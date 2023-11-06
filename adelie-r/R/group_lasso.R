
#' Group basil function
#' @export
group_basil <- function(
    X,
    y,
    groups,
    group_sizes,
    alpha=1,
    penalty=sqrt(group_sizes),
    method=ifelse(ncol(X) <= 500, 'cov', 'naive'),
    user_lmdas=double(0),
    max_n_lambdas=100,
    n_lambdas_iter=5,
    use_screen_rule=TRUE,
    do_early_exit=TRUE,
    verbose_diagnostic=FALSE,
    delta_screen_size=5,
    max_screen_size=ncol(X),
    max_n_cds=as.integer(1e5),
    tol=1e-7,
    rsq_slope_tol=1e-3,
    rsq_curv_tol=1e-3,
    newton_tol=1e-8,
    newton_max_iters=100,
    min_ratio=1e-2,
    n_threads=16
)
{
    if (method == 'cov') {
        method_f__ = group_basil_cov__
    } else if (method == 'naive') {
        method_f__ = group_basil_naive__
    } else {
        stop("Unknown method type.")
    }

    method_f__(
        X, y, as.integer(groups), as.integer(group_sizes), alpha, penalty,
        user_lmdas, max_n_lambdas, n_lambdas_iter,
        use_screen_rule, do_early_exit, verbose_diagnostic,
        delta_screen_size, max_screen_size,
        max_n_cds, tol, rsq_slope_tol, rsq_curv_tol,
        newton_tol, newton_max_iters, min_ratio, n_threads
    )
}