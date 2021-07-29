# Clear workspace
rm(list = ls())

# Load NBR
if(!require(NBR)) install.packages("NBR"); library(NBR)

# Load data from example 2
data2 <- read.csv("data/sample_input_2.csv")
tic <- Sys.time()
nbr_result <- nbr_lm(net = data2[,-(1:3)], nnodes = 116,
                     idata = data2[,1:3], mod = "~ Group + Sex * Age",
                     thrP = 0.01, nperm = 1000)
toc <- Sys.time()
print(toc-tic)
print(nbr_result$fwe)
