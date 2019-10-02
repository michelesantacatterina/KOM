##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
#*****************************************************************************************************************************
# Author: Michele Santacatterina
# Date: 24 September 2019
#
# Description: In the original manuscript, we minimize want to minimize the CMSE of the weighted estimator wrt to GATE. This quantity depends
# on unknown functions which we choose to be embedded into an RKHS (the space induced by the kernel). The kernel depends on
# some hyperparameters which we want to tune. We tune these hyperparameters by maximum likelihood gaussian processes.  To do so,
# we specify a Gaussian Process (GP) prior f with covariance identified by the product kernel K and suppose that we observed
#the potential outcome Yi(a) from f(Xi) with Gaussian noise of variance sigma2.  We then maximize the marginal likelihood
# of seeing the data with respect to the hyperparameters.
#*****************************************************************************************************************************
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


#devtools::load_all()


##############################################################################################################################
#
# The following functions are for tuning the kernel's hyperparameters with R. I personally suggest to use scikit-learn
# from python, but this is a simple reimplementation in R. I only consider Gaussian, Linear (vanilla) and polynomial kernels.
#
##############################################################################################################################


#######################################################
#######################################################
#
# Gaussian/Linear/Polynomial kernel
#
# vect: vector of initial values for hyperparameters
# x: matrix of confounders
# degree: degree of polynomial kernel
#
#######################################################
#######################################################
#Gaussian kernel
Kgauss <- function(vect,X,degree){
  X <- as.matrix(X)
  Scale <- vect[2]^2
  B <-  exp( -(Scale) * as.matrix(dist(X, upper=T, diag=T))^2 )
  return(B)
}


#Linear (vanilla) kernel
Klinear <-  function(vect,X,degree){
  X <- as.matrix(X)
  offset <- vect[2]
  B <- (X%*%t(X) + offset)
  return(B)
}

#Polynomial kernel
Kpoly <-  function(vect,X,degree){
  X <- as.matrix(X)
  offset <- vect[2]
  B <- (X%*%t(X) + offset)^degree
  return(B)
}


#######################################################
#######################################################
#
# function that returns the gram matrix
#
# vect: vector of initial values for hyperparameters
# X: matrix of confounders
# kernel: which kernel between gaussian, linear, and poly
# degree: degree of polynomial kernel
#
#######################################################
#######################################################
#function to obtain the Gram matrix based on the kernel you chose
KGram <- function(vect,X,kernel,degree){

  if(kernel=="rbf"){
    K <- Kgauss(vect,X,degree)
  }
  if(kernel=="linear"){
    K <- Klinear(vect,X,degree)
  }
  if(kernel=="poly"){
    K <- Kpoly(vect,X,degree)
  }
  return(K)

}


#######################################################
#######################################################
#
# function that returns the tune hyperparameters by
# maximizing the log likelihood.
#
# vect: vector of initial values for hyperparameters
# X: matrix of confounders
# Y: vector of outcome
# nte: sample size
# kernel: which kernel between gaussian, linear, and poly
# degree: degree of polynomial kernel
#
#######################################################
#######################################################

#minus log likelihood
#this is the actual function we want to minimize wrt to the hyperparameter
#for more info check eq. 5.8 of http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
minus_log_likelihood <- function(vect,X,Y,nte,kernel,degree){
  sn2 <- vect[3]^2
  gamma2 <- vect[1]^2
  if(kernel=="gaussian"){ K_plus <- gamma2*Kgauss(vect,X,degree) + sn2*diag(nte) }
  if(kernel=="linear"){ K_plus <- gamma2*Klinear(vect,X,degree) + sn2*diag(nte) }
  if(kernel=="poly"){ K_plus <- gamma2*Kpoly(vect,X,degree) + sn2*diag(nte) }
  K_plus_inv <-tryCatch(solve(K_plus, tol = 1e-21), error=function(e) NULL)
  if((class(K_plus_inv) != "try-error") & (is.null(K_plus_inv)!=TRUE)){
    z <- determinant(K_plus, logarithm=TRUE)
    K_plus_log_det <- as.numeric((z$sign*z$modulus)) # log-determinant of K_plus
    out <- 0.5 * ( t(Y) %*% K_plus_inv %*% Y ) + 0.5 * K_plus_log_det + (nte/2)*log(2*pi)
  }
  if((class(K_plus_inv) == "try-error") | (is.null(K_plus_inv)==TRUE)){
    out <- NULL
  }
  return(out)
}



##############################################################################################################################
#
# The following function tune the hyperparameter
# Parameters:
# outcome: the outcome uunder study (continuous)
# intervention: the treatment or intervantion that is evaluated (binary)
# confounders: matrix/data frame containing all confounders (mix)
# sample_population: vector containing info about the sample and the populations,
#             when TATE is the estimand of interest, sample_population=1 is the sample, sample_population=0 is the target population of interest
# kernel: which kernel - if gpml==R, then only gaussian, linear and poly (degree) are available. When using python, the user
#         can construct its own kernel, or use the one from scikit-learn like rbf, matern, dot, etc. More info here:
#         https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html We provide
#         the most classical one  as  basis.
# estimand: which  estimand is of interest? eg, SATE, TATE
# gpml: choose between "R" and "python" as described above (default python)
# operator: this variable is needed for python, since it allows for more complex kernels, we can choose for example, "single"
#           "sum" or "prod", for a single, sum of kernels and product of kernels repectively.
# scale: choose to center and/or scale the confounders.
#
# Some notes: remember that if you want to use python, you can choose operator <- "prod", then, kernel <- c("poly","poly") and
#             degree <- c(2,2) to get a kernel that is a product of 2 polynomial kernels of degree 2.
#
# Return: gpr from GaussianProcessRegressor, Gram matrix, and hyperparameters
#         a note on the hyperparameters: we return the hyperparameter in the following order
#         1. the first is the ConstantKernel() hyperparameter that controls the overall scale of the kernel (gamma)
#         2. the hyperparameters specific for the kernel, e.g., polynomial (parameter that controls the importance
#             of higher orders degrees), rbf (scale), etc.
#         3. the last hyperparameter is sigma^2, the Gaussian noise
#
#
##############################################################################################################################



tune <- function(outcome,
                 intervention,
                 confounders,
                 sample_population=rep(1,length(outcome)),
                 gpml="Python",
                 kernel=c("poly"),
                 degree=c(1),
                 operator="single",
                 scale=c(TRUE,TRUE),
                 python_path="~/anaconda3/bin/python"){

  tryCatch(use_python(python_path),
           error=function(e) NULL)

  #######################################################
  #######################################################
  #
  # The overall idea is to run one gaussian process
  # regression and tune the hyperparameters within each
  # treatment group (0,1)
  #
  #######################################################
  #######################################################

  #######################################################
  #******************************************************
  # Get the data in the proper way for the analysis wrt
  # estimand of interest
  #******************************************************
  #######################################################


        #scale based on what the user chose
        confounders <- scale(confounders,center = scale[1], scale = scale[2])


        #get treated (t1) and untreated (t0) in the sample population
        t1      <- as.integer(intervention[which(sample_population==1)])
        t0      <- as.integer((1-intervention[which(sample_population==1)]))

        #save for convenience treated and control data in 2 separate dataframe X1, and X0 from sample population
        Xsample <- data.frame(confounders[which(sample_population==1),],outcome[which(sample_population==1)],intervention[which(sample_population==1)])
        X1      <- Xsample[which(Xsample$intervention==1),]
        X0      <- Xsample[which(Xsample$intervention==0),]

        #save for convenience treated and control outcomes in 2 separate vectors y1, and y0.
        y       <- outcome[which(sample_population==1)]
        y1      <- y[which(Xsample$intervention==1)]
        y0      <- y[which(Xsample$intervention==0)]

        #save treated and control sizes
        n1      <- length(which(Xsample$intervention==1))
        n0      <- length(which(Xsample$intervention==0))

        # GPML optimization --- "L-BFGS-B" algorithm
        print("GPML Optimization")

        #this is the tolerance for the "L-BFGS-B" algorithm
        tol <- 1e-08


  #######################################################
  #******************************************************
  # Tune the hyperparameters
  #******************************************************
  #######################################################


        #######################################################
        # R gpml
        #######################################################

        if(gpml=="R"){

          #check that the kernels and degrees are ok
          if( ((kernel != "poly") &
             (kernel != "rbf") ) |
             (length(degree)>1)
          ){
            print("You are using GPML with R, please choose poly or rbf kernels with length(degree)==1 \\
                  Please consider using Python GPML \\
                  For now we use polynomial kernel with degree 1")
            kernel <- c("poly")
            degree <- c(1)
          }

          #we use optim to tune the hyperparameters

          #control
          system_time_GPML0 <- system.time(
          res.optim2_0 <- tryCatch(optim(par=c(1, 1, 1, 1),
                                         fn=minus_log_likelihood,
                                         method=c("L-BFGS-B"),
                                         lower = rep(tol,4),
                                         hessian=TRUE,
                                         control=list(trace=0, maxit=1000),
                                         X=X0[,1:(dim(X0)[2]-2)],
                                         Y=y0,
                                         nte=n0,
                                         kernel=kernel,
                                         degree=degree),
                                   error=function(e) NULL)
          )

          #treated
          system_time_GPML1 <- system.time(
          res.optim2_1 <- tryCatch(optim(par=c(1, 1, 1, 1),
                                         fn=minus_log_likelihood,
                                         method=c("L-BFGS-B"),
                                         lower = rep(tol,4),
                                         hessian=TRUE,
                                         control=list(trace=0, maxit=1000),
                                         X=X1[,1:(dim(X1)[2]-2)],
                                         Y=y1,
                                         nte=n1,
                                         kernel=kernel,
                                         degree=degree),
                                   error=function(e) NULL)
          )

        }#end if(gpml=="R")

        #######################################################
        # Python - sklearn gaussianprocessregressor gpml
        # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        #######################################################

        if(gpml=="Python"){

          #we need to import numpy
          np <- import("numpy")

          #create python objects for confounders and outcomes within treated and controls
          #temp steps for python objects
          mX0 <- as.matrix(X0[,1:(dim(X0)[2]-2)])
          mX1 <- as.matrix(X1[,1:(dim(X1)[2]-2)])
          mY0 <- as.matrix(y0)
          mY1 <- as.matrix(y1)

          #python np_arrays
          pyX0   <- np_array( np$array(mX0), dtype="float")
          pyX1   <- np_array( np$array(mX1), dtype="float")
          pyY0   <- np$array( mY0, dtype="float" )
          pyY1   <- np$array( mY1, dtype="float" )

          #this is the python file we will run containing all the info about gaussian process regressor
          #if you want to add more kernels or modify the existing one, this is the file you should modify.
          source_python("gp_simu_gate.py")

          #this is just a control, if the user chooses a single kernel I just need to add something
          #to degree[2]
          if(is.na(degree[2])){
            degree[2] <- 1
          }
          if(is.na(kernel[2])){
            kernel[2] <- " "
          }

          #control
          system_time_GPML0 <- system.time(
            res.optim2_0    <- tryCatch(gp(pyX0,
                                           pyY0,
                                           degree1=degree[1],
                                           degree2=degree[2],
                                           k1=kernel[1],
                                           k2=kernel[2],
                                           operator=operator),
                                        error=function(e) NULL)
          )

          #save results (tuned hyperparameters) into the list res.optim2_0$par
          #remember that gaussianprocessregressor returns log(hyperparameters), so I need to take the exp.
          res.optim2_0$par <-exp(res.optim2_0$gpr$kernel_$theta)

          #treated
          system_time_GPML1 <- system.time(
            res.optim2_1    <- tryCatch(gp(pyX1,
                                           pyY1,
                                           degree1=degree[1],
                                           degree2=degree[2],
                                           k1=kernel[1],
                                           k2=kernel[2],
                                           operator=operator),
                                        error=function(e) NULL)
          )

          #save hyperparameters for treated
          res.optim2_1$par <-exp(res.optim2_1$gpr$kernel_$theta)

        }#end if(gpml=="Python")

  #return the tuned hyperparameters
  return(list(res.optim2_0=res.optim2_0,
              res.optim2_1=res.optim2_1,
              system_time_GPML0=system_time_GPML0,
              system_time_GPML1=system_time_GPML1))

}

