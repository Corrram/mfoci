library(doBy)
library(dplyr)
library(RANN)

#-- New sample generated from single endogenuous and d=1 exogenous variables
#-- Function; input vectors x,y; output df with col y1,y2
BiPhi <- function(x,y){
  if(length(x)!=length(y)){print("Objects of different size")}
  if(length(x)==length(y)){
  n <- length(x)
  df <- data.frame(x=x, y=y, rk_x = rank(x)/(n+1), rk_y = rank(y)/(n+1))
  df <- df %>% arrange(rk_x) # sort by ranked values of x
  df[n+1,] <- df[1,]
  df <- data.frame(y1=df$rk_y,y2=lead(df$rk_y))
  df <- df[1:n,]
  }
  return(df)
}


#-- New sample generated from single endogenuous and d exogenous variables
#-- Based on dimension reduction
#-- Based on package RANN
#-- Function: input df X, vector y; output df with col y1,y2
MPhi <- function(X,y){
  X <- as.data.frame(X)
  if(nrow(X)!=length(y)){print("Objects of different size")}
  if(nrow(X)==length(y)){
  df <- data.frame(X,y)
  d <- ncol(X)
  for(i in 1:d){df[,i] <- rank(df[,i],ties.method = "random")}
  n <- nrow(df);
  df$rk_y <- rank(df[,d+1],ties.method="random")/(n+1);
  # nearest neighbor; package: RANN
  df$rk_y_nn <- rank(df$y[nn2(df[,1:d],df[,1:d],k=2)$nn.idx[,2]],ties.method = "random")/(n+1);
  df <- data.frame(y1=df$rk_y, y2=df$rk_y_nn)
  return(df)
  }
}


#-- Normed values for T, R2 and Q
#-- meth: T, R2
#-- Function: input df X, vector y; output value for T, R2 or Q
my_norm <- function(X,y){    
  MP <- MPhi(X,y)
  n <- nrow(MP)
  MP <- MP*(n+1)
  # Estimate Spearman Footrule
  TT <- 1-3/(n^2-1)*sum(abs(MP$y1-MP$y2)) + 3/(n^2-1)*((sum(MP$y2)+sum(MP$y1)-n*(n+1)))
  # Estimate Gini Gamma
  Q <- 2/(n*(n+1))*(sum(abs(n+1-MP$y1-MP$y2))-sum(abs(MP$y1-MP$y2))) + 4/(n*(n+1))*((sum(MP$y2)+sum(MP$y1)-n*(n+1)))
  # Estimate Spearman rho
  R2 <- cor(MP$y1,MP$y2, method = "spearman")
  # Normalization
  D <- data.frame(X,y)
  A <- as.data.frame(table(D[,length(D)]))
  z1 <- rank(sort(sample(x=as.numeric(A$Var1), sum(as.numeric(A$Freq)), prob=A$Freq, replace = TRUE)),ties.method="random")/(n+1)
  z2 <- rank(sort(sample(x=as.numeric(A$Var1), sum(as.numeric(A$Freq)), prob=A$Freq, replace = TRUE)),ties.method="random")/(n+1)
  Mp <- MPhi(z1,z2)
  n <- nrow(Mp)
  Mp <- Mp*(n+1)
  # Estimate Spearman Footrule
  TTmax <- 1-3/(n^2-1)*sum(abs(Mp$y1-Mp$y2)) + 3/(n^2-1)*((sum(Mp$y2)+sum(Mp$y1)-n*(n+1)))
  # Estimate Gini Gamma
  Qmax <- 2/n^2*(sum(abs(n+1-Mp$y1-Mp$y2))-sum(abs(Mp$y1-Mp$y2)))
  # Estimate Spearman rho
  R2max <- cor(Mp$y1,Mp$y2, method = "spearman")
  # Output
  Out <- data.frame(T=TT/TTmax, R2=R2/R2max, Q=Q/Qmax)
  return(Out)
}

