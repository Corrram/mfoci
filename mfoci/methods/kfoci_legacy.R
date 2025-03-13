if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
library("pacman")
p_load(KPC, proxy, kernlab)

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
kernel <- args[2]
# path <- "C:\\Users\\admin\\Documents\\repositories\\copulas_in_systemic_risk\\mfoci\\mfoci\\methods"
# kernel <- "rbfdot"
setwd(path)

num_features <- NULL
stop <- TRUE
x <- read.csv("x.csv")
y <- read.csv("y.csv")

kernel_func <- get(kernel, envir = asNamespace("kernlab"))
called_kernel <- do.call(kernel_func, list())
result <- KFOCI(y, x,  called_kernel, num_features = num_features, stop = stop, numCores = 1)
selected_cols <- colnames(x)[result]
write.csv(selected_cols, paste0(path, "/selected_cols.csv"), row.names = FALSE)
