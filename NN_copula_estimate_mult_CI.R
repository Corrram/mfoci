library(FOCI)
require(gtools)

# load my_norm
source("NN_copula_estimate.R")

my_norm_MV_CondInd <- function(X,Y){
  df <- data.frame(X,Y)
  Y <- as.data.frame(Y)
  dX <- length(as.data.frame(X))
  dY <- length(as.data.frame(Y))
  n <- nrow(df)
  #--- 
  ZW <- numeric()
  weight <- numeric()
  weight[1] <- 0
  ZW[1] <- as.numeric(my_norm(X,Y[,1])[1])
  if(dY>1){
  for(i in 2:dY){
    weight[i] <- as.numeric(my_norm(Y[,1:(i-1)],Y[,i])[1])
    ZW[i] <- as.numeric(my_norm(data.frame(X,Y[,1:(i-1)]),Y[,i])[1]) 
  }
  }
  return(data.frame(T_CondInd=1-(dY-sum(ZW))/(dY-sum(weight))))
}

my_norm_CB_MV_CondInd <- function(X,Y){
  df <- data.frame(X,Y)
  Y <- as.data.frame(Y)
  dX <- length(as.data.frame(X))
  dY <- length(as.data.frame(Y))
  n <- nrow(df)
  #--- 
  ZW <- numeric()
  weight <- numeric()
  weight[1] <- 0
  ZW[1] <- as.numeric(my_norm_CB(X,Y[,1])[1])
  if(dY>1){
    for(i in 2:dY){
      weight[i] <- as.numeric(my_norm_CB(Y[,1:(i-1)],Y[,i])[1])
      ZW[i] <- as.numeric(my_norm_CB(data.frame(X,Y[,1:(i-1)]),Y[,i])[1]) 
    }
  }
  return(data.frame(T_CondInd=1-(dY-sum(ZW))/(dY-sum(weight))))
}

CODEC_MV_CondInd <- function(X,Y){
  df <- data.frame(X,Y)
  Y <- as.data.frame(Y)
  dX <- length(as.data.frame(X))
  dY <- length(as.data.frame(Y))
  n <- nrow(df)
  #--- 
  ZW <- numeric()
  weight <- numeric()
  weight[1] <- 0
  ZW[1] <- as.numeric(codec(Y[,1],X))
  if(dY>1){
    for(i in 2:dY){
      weight[i] <- as.numeric(codec(Y[,i],Y[,1:(i-1)]))
      ZW[i] <- as.numeric(codec(Y[,i],data.frame(X,Y[,1:(i-1)])))
    }
  }
  return(data.frame(T_CondInd=1-(dY-sum(ZW))/(dY-sum(weight))))
}


CODEC_MV_CondInd_Perm2 <- function(X,Y){
  df <- data.frame(X,Y)
  Y <- as.data.frame(Y)
  dX <- length(as.data.frame(X))
  dY <- length(as.data.frame(Y))
  n <- nrow(df)
  Result <- numeric()
  perm <- permutations(n = dY, r = dY, v = 1:dY)
  perm <- perm[sample(1:factorial(dY), size = dY, replace = FALSE),]
  #--- 
  for (l in 1:nrow(perm)){
    Y <- Y[,perm[l,]]
    ZW <- numeric()
    weight <- numeric()
    weight[1] <- 0
    ZW[1] <- as.numeric(codec(Y[,1],X))
    if(dY>1){
      for(i in 2:dY){
        weight[i] <- as.numeric(codec(Y[,i],Y[,1:(i-1)]))
        ZW[i] <- as.numeric(codec(Y[,i],data.frame(X,Y[,1:(i-1)])))
      }
    }
  Result[l] <- 1-(dY-sum(ZW))/(dY-sum(weight))
  }
  return(data.frame(T_CondInd=mean(Result)))
}