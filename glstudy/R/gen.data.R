# Common routines for data generation that is useful.

#' Generates standard normal data X size nxp. 
#' If cov is provided, it is used to scale each row of X.
#' Coefficients are generated from uniform(0,1).
#' If sparsity is > 0, then some entries of beta are zero-ed out.
#' If n.groups is provided, then n.groups number of groups are randomly generated.
#' The groups are returned in 0-index!
#' @export
gen.data <- function(n, p, cov=NA, sparsity=0, n.groups=NA, seed=NA)
{
    if (!is.na(seed)) set.seed(seed)
    X <- matrix(rnorm(n * p), n, p)
    if (!is.na(cov)) {
        R <- chol(cov)
        X <- X %*% R
    }
    beta <- runif(p) * rbinom(1, p, 1-sparsity)
    y <- X %*% beta + rnorm(n)
    A <- (t(X) %*% X) / n
    r <- as.numeric(t(X) %*% y / n)
    
    # generate groups
    if (!is.na(n.groups)) {
        groups <- c(0, sort(sample.int(p-1, n.groups-1, replace=FALSE)), p)
        group.sizes <- groups[2:(n.groups+1)] - groups[1:n.groups]
        groups <- as.integer(groups[1:n.groups])
        group.sizes <- as.integer(group.sizes)
    }
    
    list(
        X=X,
        y=y,
        beta=beta,
        A=A,
        r=r,
        groups=groups,
        group.sizes=group.sizes
    )
}