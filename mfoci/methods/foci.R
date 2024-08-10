if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
library("pacman")
p_load(KPC, proxy, kernlab, FOCI)
#############################################
# HELPER FUNCTIONS for codec
#
# .estimateConditionalQ
# .estimateConditionalS
# .estimateConditionalT
# .estimateQ
# .estimateS
# .estimateT
##############################################


# codec -------------------------------------------------------------------------
#' Estimate the conditional dependence coefficient (CODEC)
#'
#' The conditional dependence coefficient (CODEC) is a measure of the amount of conditional dependence between
#' a random variable Y and a random vector Z given a random vector X, based on an i.i.d. sample of (Y, Z, X).
#' The coefficient is asymptotically guaranteed to be between 0 and 1.
#'
#' @param Y Vector (length n)
#' @param Z Matrix (n by q)
#' @param X Matrix (n by p), default is NULL
#' @param na.rm Remove NAs if TRUE
#' @details The value returned by codec can be positive or negative. Asymptotically, it is guaranteed
#' to be between 0 and 1. A small value indicates low conditional dependence between Y and Z given X, and
#' a high value indicates strong conditional dependence. The codec function is used by the  \code{\link{foci}} function
#' for variable selection.
#' @return The conditional dependence coefficient (CODEC) of Y and Z given X. If X == NULL, this is just a
#' measure of the dependence between Y and Z.
#' @import data.table
#' @importFrom stats complete.cases sd
#' @export
#' @author Mona Azadkia, Sourav Chatterjee, Norman Matloff
#' @references Azadkia, M. and Chatterjee, S. (2019). A simple measure
#' of conditional dependence.
#' \url{https://arxiv.org/pdf/1910.12327.pdf}.
#' @seealso \code{\link{foci}}, \code{\link[XICOR]{xicor}}
#' @examples
#' n = 1000
#' x <- matrix(runif(n * 2), nrow = n)
#' y <- (x[, 1] + x[, 2]) %% 1
#' # given x[, 1], y is a function of x[, 2]
#' codec(y, x[, 2], x[, 1])
#' # y is a function of x
#' codec(y, x)
#' z <- rnorm(n)
#' # y is a function of x given z
#' codec(y, x, z)
#' # y is independent of z given x
#' codec(y, z, x)
codec <- function(Y, Z, X = NULL, na.rm = TRUE){
  if(is.null(X)) {
    # if inputs are not in proper matrix format change if possible
    # otherwise send error
    if(!is.vector(Y)) {
      Y <- as.vector(Y)
    }
    if(!is.matrix(Z)) {
      Z <- as.matrix(Z)
    }
    if((length(Y) != nrow(Z))) stop("Number of rows of Y and X should be equal.")
    if (na.rm == TRUE) {
      # NAs are removed here:
      ok <- complete.cases(Y, Z)
      Z <- as.matrix(Z[ok,])
      Y <- Y[ok]
    }

    n <- length(Y)
    if(n < 2) stop("Number of rows with no NAs should be bigger than 1.")

    estimateTResult <- .estimateT(Y, Z)
    return(estimateTResult)
  }
  # if inputs are not in proper matrix format change if possible
  # otherwise send error
  if(!is.vector(Y)) {
    Y <- as.vector(Y)
  }
  if(!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if(!is.matrix(Z)) {
    Z <- as.matrix(Z)
  }
  if((length(Y) != nrow(X))) stop("Number of rows of Y and X should be equal.")
  if((length(Y) != nrow(Z))) stop("Number of rows of Y and Z should be equal.")
  if((nrow(Z) != nrow(X))) stop("Number of rows of Z and X should be equal.")
  if (na.rm == TRUE) {
    # NAs are removed here:
    ok <- complete.cases(Y,Z,X)
    Z <- as.matrix(Z[ok,])
    Y <- Y[ok]
    X <- as.matrix(X[ok,])
  }

  n <- length(Y)
  if(n < 2) stop("Number of rows with no NAs should be bigger than 1.")

  estimateConditionalTResult <- estimateConditionalT(Y, Z, X)
  return(estimateConditionalTResult)
}


# .estimateConditionalQ -------------------------------------------------------------------------
# Estimate Q(Y, Z | X)
#
# Estimate Q(Y, Z | X), the numinator of the measure of conditional dependence of Y on Z given X
#
# @param X: Matrix of predictors (n by p)
# @param Z: Matrix of predictors (n by q)
# @param Y: Vector (length n)
#
# @return estimation \eqn{Q(Y, Z|X)}
.estimateConditionalQ <- function (Y, X, Z) {

  id <- group <- rnn <- NULL

  if(!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if(!is.matrix(Z)) {
    Z <- as.matrix(Z)
  }

  n <- length(Y)

  W <- cbind(X, Z)

  # compute the nearest neighbor of X
  nn_X <- RANN::nn2(X, query = X, k = 3)
  nn_index_X <- nn_X$nn.idx[, 2]
  # handling repeated data
  repeat_data <- which(nn_X$nn.dists[, 2] == 0)

  df_X <- data.table::data.table(id = repeat_data, group = nn_X$nn.idx[repeat_data, 1])
  df_X[, rnn := .randomNN(id), by = "group"]

  nn_index_X[repeat_data] <- df_X$rnn
  # nearest neighbors with ties
  ties <- which(nn_X$nn.dists[, 2] == nn_X$nn.dists[, 3])
  ties <- setdiff(ties, repeat_data)

  if(length(ties) > 0) {
    helper_ties <- function(a) {
      distances <- proxy::dist(matrix(X[a, ], ncol = ncol(X)), matrix(X[-a, ], ncol = ncol(X)))
      ids <- which(distances == min(distances))
      x <- sample(ids, 1)
      return(x + (x >= a))
    }

    nn_index_X[ties] <- sapply(ties, helper_ties)
  }

  # compute the nearest neighbor of W
  nn_W <- RANN::nn2(W, query = W, k = 3)
  nn_index_W <- nn_W$nn.idx[, 2]
  repeat_data <- which(nn_W$nn.dists[, 2] == 0)

  df_W <- data.table::data.table(id = repeat_data, group = nn_W$nn.idx[repeat_data])
  df_W[, rnn := .randomNN(id), by = "group"]

  nn_index_W[repeat_data] <- df_W$rnn
  # nearest neighbors with ties
  ties <- which(nn_W$nn.dists[, 2] == nn_W$nn.dists[, 3])
  ties <- setdiff(ties, repeat_data)

  if(length(ties) > 0) {
    helper_ties <- function(a) {
      distances <- proxy::dist(matrix(X[a, ], ncol = ncol(X)), matrix(X[-a, ], ncol = ncol(X)))
      ids <- which(distances == min(distances))
      x <- sample(ids, 1)
      return(x + (x >= a))
    }

    nn_index_W[ties] <- sapply(ties, helper_ties)
  }

  # estimate Q
  R_Y <- rank(Y, ties.method = "max")
  Q_n <- sum(apply(rbind(R_Y, R_Y[nn_index_W]), 2, min),
            -apply(rbind(R_Y, R_Y[nn_index_X]), 2, min)) / (n^2)
  return(Q_n)
}




# .estimateConditionalS -------------------------------------------------------------------------
# Estimate S(Y, X)
#
# Estimate S(Y, X), the denuminator of the measure of dependence of Y on Z given X
#
# @param X: Matrix of predictors (n by p)
# @param Y: Vector (length n)
#
# @return estimation \eqn{S(Y, X)}
.estimateConditionalS <- function (Y, X){

  id <- group <- rnn <- NULL

  if(!is.matrix(X)) {
    X <- as.matrix(X)
  }
  n <- length(Y)

  # compute the nearest neighbor of X
  nn_X <- RANN::nn2(X, query = X, k = 3)
  nn_index_X <- nn_X$nn.idx[, 2]
  repeat_data <- which(nn_X$nn.dists[, 2] == 0)

  df_X <- data.table::data.table(id = repeat_data, group = nn_X$nn.idx[repeat_data, 1])
  df_X[, rnn := .randomNN(id), by = "group"]

  nn_index_X[repeat_data] <- df_X$rnn
  # nearest neighbors with ties
  ties <- which(nn_X$nn.dists[, 2] == nn_X$nn.dists[, 3])
  ties <- setdiff(ties, repeat_data)

  if(length(ties) > 0) {
    helper_ties <- function(a) {
      distances <- proxy::dist(matrix(X[a, ], ncol = ncol(X)), matrix(X[-a, ], ncol = ncol(X)))
      ids <- which(distances == min(distances))
      x <- sample(ids, 1)
      return(x + (x >= a))
    }

    nn_index_X[ties] <- sapply(ties, helper_ties)
  }

  # estimate S
  R_Y <- rank(Y, ties.method = "max")
  S_n <- sum(R_Y - apply(rbind(R_Y, R_Y[nn_index_X]), 2, min)) / (n^2)

  return(S_n)
}


# estimateConditionalT -------------------------------------------------------------------------
# Estimate T(Y, Z | X)
#
# Estimate T(Y, Z | X), the measure of dependence of Y on Z given X
#
# @param Y: Vector (length n)
# @param Z: Matrix of predictors (n by q)
# @param X: Matrix of predictors (n by p)
#
# @return estimation of \eqn{T(Y, Z|X)}.
.estimateConditionalT <- function(Y, Z, X){
  S <- .estimateConditionalS(Y, X)

  # happens only if Y is constant
  if (S == 0) {
    return(1)
  } else {
    return(.estimateConditionalQ(Y, X, Z) / S)
  }
}




# .estimateQ -------------------------------------------------------------------------
# Estimate Q(Y, X)
#
# Estimate Q(Y, X), the numinator of the measure of dependence of Y on X
#
# @param X: Matrix of predictors (n by p).
# @param Y: Vector (length n).
#
# @return estimation of \eqn{Q(Y, X)}.
.estimateQ <- function(Y, X) {

  id <- group <- rnn <- NULL

  if(!is.matrix(X)) {
    X <- as.matrix(X)
  }

  n <- length(Y)
  nn_X <- RANN::nn2(X, query = X, k = 3)
  # remove the first nearest neighbor for each x which is x itself in case of no repeat data
  # when there is repeated data this is wrong but for repeated data we find the nearest
  # neighbors separately.
  nn_index_X <- nn_X$nn.idx[, 2]

  # find all data points that are not unique
  repeat_data <- which(nn_X$nn.dists[, 2] == 0)

  # for the repeated data points, choose one of their identicals at random and set its index
  # as the index of the nearest neighbor
  df_X <- data.table::data.table(id = repeat_data, group = nn_X$nn.idx[repeat_data, 1])
  df_X[, rnn := .randomNN(id), by = "group"]
  nn_index_X[repeat_data] <- df_X$rnn

  # nearest neighbors with ties
  ties <- which(nn_X$nn.dists[, 2] == nn_X$nn.dists[, 3])
  ties <- setdiff(ties, repeat_data)
  if(length(ties) > 0) {
    helper_ties <- function(a) {
      distances <- proxy::dist(matrix(X[a, ], ncol = ncol(X)), matrix(X[-a, ], ncol = ncol(X)))
      ids <- which(distances == min(distances))
      x <- sample(ids, 1)
      return(x + (x >= a))
    }

    sapplyResult <- sapply(ties, helper_ties)
    nn_index_X[ties] <- sapplyResult
  }


  # estimate Q
  R_Y <- rank(Y, ties.method = "max")
  L_Y <- rank(-Y, ties.method = "max")
  rbindResult <- rbind(R_Y, R_Y[nn_index_X])
  applyResult <- apply(rbindResult, 2, min)
  Q_n <- sum(applyResult - (L_Y^2)/n) / (n^2)

  return(Q_n)
}



# .estimateS -------------------------------------------------------------------------
# Estimate S(Y)
#
# Estimate S(Y) , the denuminator of the measure of dependence of Y on X
#
# @param Y: Vector (length n).
# @return estimation of \eqn{S(Y)}.
.estimateS <- function (Y) {
  n <- length(Y)
  L_Y <- rank(-Y, ties.method = "max")
  S_n <- gmp::asNumeric(sum(gmp::as.bigz(L_Y) * gmp::as.bigz(n - L_Y))) / (n^3)
  return(S_n)
}



# .estimateT -------------------------------------------------------------------------
# Estimate T(Y, X)
#
# Estimate T(Y, X), the measure of dependence of Y on X
#
# @param X: Matrix of predictors (n by p)
# @param Y: Vector (length n)
# @return estimation of \eqn{T(Y, X) <- Q(Y, X) / S(Y)}.
.estimateT <- function(Y, X){
  # May 15, Mona removed FOCI:::
  S <- .estimateS(Y)
  # happens only if Y is a constant vector.
  if (S == 0) {
    return(1)
  } else {
    q <- .estimateQ(Y, X)
    return(q / S)
  }
}

.randomNN <- function(ids) {
  m <- length(ids)

  x <- sample(x = (m - 1), m, replace = TRUE)
  x <- x + (x >= (1:m))

  return(ids[x])
}

df <- read.csv("clayton_data.csv")
Y <- df$x
Z <- df$z
result <- codec(Y, Z)

Y <- c(10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5)
Z <- c(8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68)
#  <- FOCI::codec(Y, Z)
result_orig_1 <- FOCI::codec(Y, Z)
result_orig_2 <- FOCI::codec(Z, Y)
result <- codec(Y, Z)
result_2 <- codec(Z, Y)
print(result)