# load my_norm
source("NN_copula_estimate.R")
source("NN_copula_estimate_mult_CI.R")

#-- Hierarchical Feature Selection, response y
#-- Hierarchical influence of exogenuous X on response y
#-- Function; input df X, vector y, number of features, method (1: "Codec", 2: "Copula T", 3: "Codec TPerm"); output list
my_FS_Mult_Hierarch <- function(X,Y,num_features,meth){
  # Basics
  df <- data.frame(X,Y)
  X <- as.data.frame(X)
  Y <- as.data.frame(Y)
  dX <- length(X)
  dY <- length(Y)
  n <- nrow(df)
  l <- min(dX,num_features)
  Var_d <- colnames(X)
  Idx <- c("Codec T", "Copula T", "Codec TPerm")
  print(paste("Hierarchical Feature Selection basen on", Idx[meth]))
  #
  H <- data.frame(Feature=rep(0,l),number=rep(0,l))
  colnames(H)[2] <- Idx[meth] 
  VarEx <- character()
  Z <- data.frame()
  if(meth==1){for(l2 in 1:l){
      h1 <- data.frame(feat=Var_d,number=0)
      if(length(Z)>0){for(l1 in 1:length(Var_d)){h1[l1,2] <- CODEC_MV_CondInd(cbind(Z,X[,Var_d[l1]]),Y)};
        H[l2,] <- h1[which.max(h1$number),];
        VarEx <- H[l2,1]; 
        Z <- cbind(Z,X[,VarEx])} else{
          for(l1 in 1:length(Var_d)){h1[l1,2] <- CODEC_MV_CondInd(X[,Var_d[l1]],Y)};
          H[l2,] <- h1[which.max(h1$number),];    
          VarEx <- H[l2,1];
          Z <- data.frame(X[,VarEx])}
      if(length(Var_d)>0) Var_d <- Var_d[-which(Var_d==H[l2,1])]
      if(length(Var_d)>1) X <- X[,Var_d]
      if(length(Var_d)==1) X <- as.data.frame(X[,Var_d]) ;colnames(X) <- Var_d
  }}
  if(meth==2){for(l2 in 1:l){
    h1 <- data.frame(feat=Var_d,number=0)
    if(length(Z)>0){for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm_MV_CondInd(cbind(Z,X[,Var_d[l1]]),Y)};
      H[l2,] <- h1[which.max(h1$number),];
      VarEx <- H[l2,1]; 
      Z <- cbind(Z,X[,VarEx])} else{
        for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm_MV_CondInd(X[,Var_d[l1]],Y)};
        H[l2,] <- h1[which.max(h1$number),];    
        VarEx <- H[l2,1];
        Z <- data.frame(X[,VarEx])}
    if(length(Var_d)>0) Var_d <- Var_d[-which(Var_d==H[l2,1])]
    if(length(Var_d)>1) X <- X[,Var_d]
    if(length(Var_d)==1) X <- as.data.frame(X[,Var_d]) ;colnames(X) <- Var_d
  }}
  if(meth==3){for(l2 in 1:l){
    h1 <- data.frame(feat=Var_d,number=0)
    if(length(Z)>0){for(l1 in 1:length(Var_d)){h1[l1,2] <- CODEC_MV_CondInd_Perm2(cbind(Z,X[,Var_d[l1]]),Y)};
      H[l2,] <- h1[which.max(h1$number),];
      VarEx <- H[l2,1]; 
      Z <- cbind(Z,X[,VarEx])} else{
        for(l1 in 1:length(Var_d)){h1[l1,2] <- CODEC_MV_CondInd_Perm2(X[,Var_d[l1]],Y)};
        H[l2,] <- h1[which.max(h1$number),];    
        VarEx <- H[l2,1];
        Z <- data.frame(X[,VarEx])}
    if(length(Var_d)>0) Var_d <- Var_d[-which(Var_d==H[l2,1])]
    if(length(Var_d)>1) X <- X[,Var_d]
    if(length(Var_d)==1) X <- as.data.frame(X[,Var_d]) ;colnames(X) <- Var_d
  }}
  if(meth==4){for(l2 in 1:l){
    h1 <- data.frame(feat=Var_d,number=0)
    if(length(Z)>0){for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm_CB_MV_CondInd(cbind(Z,X[,Var_d[l1]]),Y)};
      H[l2,] <- h1[which.max(h1$number),];
      VarEx <- H[l2,1]; 
      Z <- cbind(Z,X[,VarEx])} else{
        for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm_CB_MV_CondInd(X[,Var_d[l1]],Y)};
        H[l2,] <- h1[which.max(h1$number),];    
        VarEx <- H[l2,1];
        Z <- data.frame(X[,VarEx])}
    if(length(Var_d)>0) Var_d <- Var_d[-which(Var_d==H[l2,1])]
    if(length(Var_d)>1) X <- X[,Var_d]
    if(length(Var_d)==1) X <- as.data.frame(X[,Var_d]) ;colnames(X) <- Var_d
  }}
  rownames(H) <- 1:nrow(H)
  H <- H[,c(2,1)]
  return(H)
}


#-- Hierarchical Feature Selection, response y
#-- Hierarchical influence of exogenuous X on response y
#-- Function; input df X, vector y, number of features, method (1: "Chatterjee", 2: "CopulaCor", 3: "ReflInvariance"); output list
my_FS_Hierarch <- function(X,y, num_features,meth){
  l <- min(length(X),num_features)
  Var_d <- colnames(X)
  Idx <- c("Chatterjee T", "CopulaCor R2", "ReflInvariance Q")
  print(paste("Hierarchical Feature Selection basen on", Idx[meth]))
  H <- data.frame(Feature=rep(l,0),number=rep(l,0))
  colnames(H)[2] <- Idx[meth] 
  VarEx <- character()
  Y <- data.frame()
  for(l2 in 1:l){
    h1 <- data.frame(feat=Var_d,number=0)
    if(length(Y)>0){for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm(data.frame(Y,X[,Var_d[l1]]),y)[meth]};
      H[l2,] <- h1[which.max(h1$number),];
      VarEx <- H[l2,1]; 
      Y <- cbind(Y,X[,VarEx])} else{
        for(l1 in 1:length(Var_d)){h1[l1,2] <- my_norm(X[,Var_d[l1]],y)[meth]};
        H[l2,] <- h1[which.max(h1$number),];    
        VarEx <- H[l2,1];
        Y <- data.frame(X[,VarEx])}
    if(length(Var_d)>0) Var_d <- Var_d[-which(Var_d==H[l2,1])]
    if(length(Var_d)>1) X <- X[,Var_d]
    if(length(Var_d)==1) X <- as.data.frame(X[,Var_d]) ;colnames(X) <- Var_d
  }
  rownames(H) <- 1:nrow(H)
  H <- H[,c(2,1)]
  return(H)
}


#-- Single Feature Selection
#-- Single individual influence of exogenuous X on response y
#-- Function; input df X, vector y, number of features, method (1: "Chatterjee", 2: "CopulaCor", 3: "ReflInvariance"); output list
my_FS_Single <- function(X,y,num_features,meth){
  l <- min(length(X),num_features)
  Feat <- character()
  Var <- colnames(X)
  Idx <- c("Chatterjee T", "CopulaCor R2", "ReflInvariance Q")
  print(paste("Single Feature Selection basen on", Idx[meth]))
  H <- data.frame(Feature=rep(l,0),number=rep(l,0))
  colnames(H)[2] <- Idx[meth]
  VarAdd <- character()
  for(l2 in 1:l){
    h1 <- data.frame(feat=Var,number=0)
    for(l1 in 1:length(Var)){
      h1[l1,2] <- my_norm(X[,Var[l1]],y)[meth] 
    }
    H[l2,] <- h1[which.max(h1$number),]
    Var <- Var[-which(Var==H[l2,1])]
    VarEx <- H[l2,1]
  }
  H
  return(H)
}


###Test
#load("bio.RData")
#X <- bio[,1:11]
#y <- bio[,12]
#my_FS_Single(X,y,4,3)
#my_FS_Hierarch(X,y,3,1)