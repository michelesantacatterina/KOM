#' Optimal Estimation of Generalized Average Treatment Effects using Kernel Optimal Matching
#'
#' \code{KOM} estimates weights that minimize the worst-case Conditional Mean Squared Error (CMSE) of the weighted
#' estimator in estimating Generalized Average Treatment Effects (GATE) over the space of weights. The manuscript
#' can be found here: \href{https://arxiv.org/pdf/1908.04748.pdf}{https://arxiv.org/pdf/1908.04748.pdf}.
#'
#'
#' We highly suggest to use \code{Python} and \code{Gurobi} for \code{KOM}. To install \code{Python}, please install
#' \href{https://www.anaconda.com/distribution/}{Anaconda}. Once Anaconda is installed please install \code{sklearn} if not already installed.
#' To install the \code{R} interface of \code{Gurobi} please follow instructions
#' \href{https://cran.r-project.org/web/packages/prioritizr/vignettes/gurobi_installation.html}{here}. \code{Gurobi} provides
#' \href{https://www.gurobi.com/downloads/end-user-license-agreement-academic/}{free academic licenses}
#' and \href{https://www.gurobi.com/downloads/eula-academic-online-course-license-request/}{free academic online course licenses}.
#'
#' @param outcome outcome under study.
#' @param intervention binary intervention/treatment.
#' @param confounders a matrix containing all confounders. Each column represents a confounder.
#' @param sample_population vector containing information about the sample population. When \code{estimand="TATE"} (see below, parameter \code{estimand}) is the estimand of interest, \code{sample_population=rep(1,n)} is the sample,
#' \code{sample_population=rep(0,nt)} is the target population of interest where nt is the sample size in the target population.
#' @param gpml specifies the method for Gaussian Process Maximum Likelihood for tuning the hyperparameters. Choices are \code{R} and \code{Python}. Default \code{gpml="Python"}.
#' @param kernel specifies the kernel to be used, choices are #' \enumerate{ \item for \code{gpml="R"}: "poly" (polynomial kernel), "rbf" (rbf kernel). Default "poly".
#' \item for \code{gpml="Python"}: "poly" (polynomial kernel), "rbf" (rbf kernel), "matern1.5" (Matern kernel with nu=1.5), and "matern2.5" (Matern kernel with nu=2.5).}
#' Default \code{kernel="poly"}.
#' @param degree degree of the polynomial kernel, e.g., 1=linear, 2=quadratic, 3=cubic. Default \code{degree=c(1)}.
#' @param operator kernel(s)' operator, choices are \code{single}, \code{sum} and \code{prod}. If \code{sum} or \code{prod} is chosen, please provide a vector of kernels and degrees for the
#' parameters \code{kernel} and \code{degree} respectively, e.g., \code{operator="prod"}, then set \code{kernel=c("poly","poly")}, \code{degree=c(2,2)} for a product of polynomial degree 2 kernels.
#' @param scale center and scale parameters for scaling/centering covariates. First parameter is center, second parameter is scale. Default \code{scale=c(TRUE,TRUE)}.
#' @param penal penalization parameter for the variance of the estimator, \eqn{\lambda}. Default \code{penal=1} - optimal, i.e., \eqn{\gamma^2/\sigma^2}.
#' @param estimand the estimand of interest, choices are between SATE, KOWATE, KOSATE, and TATE. Default \code{estimand="SATE"}.
#' @param weights weights V in the main manuscript. Then we compute SATE in a population pre-specified by a set of weights. For example truncated or overlap weights. In other words,
#' we want to compute the set of weights that minimize the conditional mean square error of the weighted estimator for SATE in a population defined by the pre-specified weights.
#' If a set of weights all equal to 1 is provided, we estimate SATE in the sample population. Default \code{weights=rep(1,length(outcome)}, where n is the sample size.
#' @param truncation_level truncation level for computing KOSATE. Default \code{truncation_level=c(0.9)}.
#' @param Presolve presolve parameter for gurobi optimizer. More info \href{https://www.gurobi.com/documentation/8.1/refman/presolve.html}{here}. Default \code{Presolve=c(2)} - aggressive.
#' @param OutputFlag enables or disables solver output. More info \href{https://www.gurobi.com/documentation/8.1/refman/outoutflag.html}{here}. Default \code{OutputFlag=c(0)}.
#' @param python_path python path. Default \code{python_path = "~/anaconda3/bin/python"}.
#' @return A list the following.
#' \item{res}{\code{gurobi} type object}
#' \item{WV}{combined vector of weights W and V size 2*n solution to the optimization problem, when \code{estimand="KOWATE" or "KOSATE"}}
#' \item{W}{vector of weights W size n solution to the optimization problem.}
#' \item{V}{vector of weights V size n solution to the optimization problem when \code{estimand="KOWATE" or "KOSATE"}.}
#' \item{res.optim2_1_par}{vector of tuned hyperparameters for treated. The first is \eqn{\gamma^2}, the hyperparameter that controls the overall scale of the kernel.
#' The last is \eqn{\gamma^2}, the Gaussian Process variance. The remaining hyperparameters are kernel specific. For example, for polynomial kernels, \eqn{\sigma}
#' controls the importance of higher orders degrees. These hyperparameters are not of direct interest of the user, they are only used to compute the Gram matrices.}
#' \item{res.optim2_0_par}{vector of tuned hyperparameters for controls. Same info as for the treated hyperparameters.}
#' \item{sigma0}{Gaussian Process variance \eqn{\gamma^2} for controls.}
#' \item{sigma1}{Gaussian Process variance \eqn{\gamma^2} for treated.}
#' \item{system_time_GPML0}{return CPU (and other) times for gpml among controls.}
#' \item{system_time_GPML1}{return CPU (and other) times for gpml among treated.}
#' \item{system_time_gurobi}{return CPU (and other) times for gurobi.}
#' \item{system_time_matrices}{return CPU (and other) times for building the matrices.}
#' \item{status}{return 1 if no errors are encountered, 0 otherwise.}
#' @author Michele Santacatterina, \email{santacatterina@cornell.edu}
#' @references Kallus, N., Santacatterina, M., "Optimal Estimation of Generalized Average
#' Treatment Effects using Kernel Optimal Matching", \url{https://arxiv.org/pdf/1908.04748.pdf}
#' @examples
#' \dontrun{
#'library(sandwich)
#'
#'set.seed(1)
#'
#'###########################################################################
#'#SATE and OWATE
#'
#'#Data Generation
#'n             <- 300
#'
#'X1            <- rnorm(n,0.1,1)
#'X2            <- rnorm(n,0.1,1)
#'
#'dta           <- data.frame(X1,X2)
#'colnames(dta) <- c("X1","X2")
#'
#'B             <- matrix(c(2,2),ncol=1)
#'
#'prt           <- 1/(1+exp( -(dta$X1 + dta$X2) ))
#'
#'Tr            <- rbinom(n,1,prt)
#'prop.table(table(Tr))
#'
#'dta           <- data.frame(dta,Tr)
#'dta           <- dta[order(dta$Tr,decreasing=TRUE),]
#'colnames(dta) <- c("X1","X2","Tr")
#'
#'effect_t      <- 2
#'
#'B             <- matrix(c(2,2),ncol=1)
#'
#'Y0            <- B[1]*dta$X1 + B[2]*dta$X2 + rnorm(n)
#'Y1            <- Y0 + effect_t
#'Y             <- Y0*(1-dta$Tr) + Y1*(dta$Tr)
#'
#'dta           <- data.frame(Y,dta$X1,dta$X2,dta$Tr)
#'colnames(dta) <- c("Y","X1","X2","Tr")
#'
#'Z             <- cbind(dta$X1,dta$X2)
#'
#'colnames(Z)   <- c("Z1","Z2")
#'dta           <- data.frame(dta,Z)
#'
#'
#'#************************************
#'#Setup
#'confounders   <- cbind(dta$Z1,dta$Z2)
#'intervention  <- dta$Tr
#'outcome       <- dta$Y
#'kernel        <- c("poly")
#'degree        <- 1
#'penal         <- 1
#'operator      <- "single"
#'
#'
#'#************************************
#'#SATE
#'estimand      <- "SATE"
#'
#'resultKOMSATE <- KOM(outcome,
#'                     intervention,
#'                     confounders,
#'                     kernel=kernel,
#'                     degree=degree,
#'                     operator=operator,
#'                     estimand=estimand)
#'
#'summary(resultKOMSATE$W)
#'lmKOMSATE <- lm(Y~Tr,data=dta,weights=resultKOMSATE$W)
#'summary(lmKOMSATE)
#'
#'
#'#************************************
#'#KOWATE
#'estimand      <- "KOWATE"
#'
#'resultKOWATE <- KOM(outcome,
#'                     intervention,
#'                     confounders,
#'                     kernel=kernel,
#'                     degree=degree,
#'                     operator=operator,
#'                     estimand=estimand)
#'
#'summary(resultKOWATE$W)
#'lmKOWATE <- lm(Y~Tr,data=dta,weights=resultKOWATE$W)
#'summary(lmKOWATE)
#'
#'
#'#************************************
#'#KOSATE
#'estimand      <- "KOSATE"
#'
#'
#'resultKOSATE <- KOM(outcome,
#'                     intervention,
#'                     confounders,
#'                     kernel=kernel,
#'                     degree=degree,
#'                     operator=operator,
#'                     estimand=estimand)
#'
#'summary(resultKOSATE$W)
#'lmKOSATE <- lm(Y~Tr,data=dta,weights=resultKOSATE$W)
#'summary(lmKOSATE)
#'
#'
#'
#'#************************************
#'#SATE with fixed V - Overlap
#'estimand      <- "SATE"
#'
#'#overlap weights
#'fit_tr        <- glm(Tr  ~ Z1 + Z2, data=dta, family="binomial")
#'pr1           <- fit_tr$fitted.values
#'overlap_w     <- dta$Tr*(1-pr1) + (1-dta$Tr)*pr1
#'
#'resultSATE_O <- KOM(outcome,
#'                     intervention,
#'                     confounders,
#'                     kernel=kernel,
#'                     degree=degree,
#'                     operator=operator,
#'                     estimand=estimand,
#'                     weights=overlap_w)
#'
#'summary(resultSATE_O$W)
#'lmSATE_O <- lm(Y~Tr,data=dta,weights=resultSATE_O$W)
#'summary(lmSATE_O)
#'
#'
#'
#'#************************************
#'#SATE with fixed V - Truncated 0.1-0.9
#'estimand      <- "SATE"
#'
#'#truncated weights
#'fit_tr        <- glm(Tr  ~ Z1 + Z2, data=dta, family="binomial")
#'pr1           <- fit_tr$fitted.values
#'overlap_w     <- dta$Tr*(1-pr1) + (1-dta$Tr)*pr1
#'indicator0109 <- (pr1>quantile(pr1,0.1))*(pr1<quantile(pr1,0.9))
#'truncated_ipw <- indicator0109*( dta$Tr/pr1 + (1-dta$Tr)/(1-pr1)  )
#'weights       <- truncated_ipw
#'
#'resultSATE_T <- KOM(outcome,
#'                     intervention,
#'                     confounders,
#'                     kernel=kernel,
#'                     degree=degree,
#'                     operator=operator,
#'                     estimand=estimand,
#'                     weights=truncated_ipw)
#'
#'summary(resultSATE_T$W)
#'lmSATE_T <- lm(Y~Tr,data=dta,weights=resultSATE_T$W)
#'summary(lmSATE_T)
#'
#'
#'#####'Compare results
#'
#'summary(resultKOMSATE$W)
#'summary(resultKOWATE$W)
#'summary(resultSATE_O$W)
#'summary(resultSATE_T$W)
#'summary(resultKOSATE$W)
#'
#'summary(lm(Y~Tr,data=dta))$coef[2]
#'summary(lmKOMSATE)$coef[2]
#'summary(lmKOWATE)$coef[2]
#'summary(lmKOSATE)$coef[2]
#'summary(lmSATE_O)$coef[2]
#'summary(lmSATE_T)$coef[2]
#'
#'sqrt(diag(sandwich(lmKOMSATE)))[2]
#'sqrt(diag(sandwich(lmKOWATE)))[2]
#'sqrt(diag(sandwich(lmKOSATE)))[2]
#'sqrt(diag(sandwich(lmSATE_O)))[2]
#'sqrt(diag(sandwich(lmSATE_T)))[2]
#'
#'
#' }
#'
#devtools::load_all()



KOM <- function(outcome,
                intervention,
                confounders,
                sample_population=rep(1,length(outcome)),
                gpml="Python",
                kernel=c("poly"),
                degree=c(1),
                operator="single",
                scale=c(TRUE,TRUE),
                estimand=c("SATE"),
                penal=c(1),
                weights=rep(1,length(outcome)),
                truncation_level=c(0.9),
                Presolve=c(2),
                OutputFlag=c(0),
                python_path="~/anaconda3/bin/python"
){

  ##############################################################################################################
  ##############################################################################################################
  #*************************************************************************************************************
  #Tune hyperparameters
  #*************************************************************************************************************
  ##############################################################################################################
  ##############################################################################################################
  tune_hyper <- tune(outcome,
                     intervention,
                     confounders,
                     sample_population,
                     gpml,
                     kernel,
                     degree,
                     operator,
                     scale,
                     python_path)

  #variables setup
  temp_intervention <- intervention[which((is.na(intervention)!=TRUE)&(sample_population==1))]
  t1                <- as.integer(temp_intervention)
  t0                <- as.integer((1-temp_intervention))
  n                 <- length(intervention)

  res <- NULL

  #save tuned hyperparameters for treated (1) and controls (0)
  res.optim2_0 <- tune_hyper$res.optim2_0
  res.optim2_1 <- tune_hyper$res.optim2_1

  #if the tuning was succesfull, build the matrices and solve the QP
  if ( (class(res.optim2_0) != "try-error") &
       (class(res.optim2_1) != "try-error") &
       (is.null(res.optim2_0)!=TRUE) &
       (is.null(res.optim2_1)!=TRUE)
  ){

    ##############################################################################################################
    ##############################################################################################################
    #*************************************************************************************************************
    #KOWATE
    #*************************************************************************************************************
    ##############################################################################################################
    ##############################################################################################################

    if(estimand=="KOWATE"){

      #######################################################
      #######################################################
      #Building matrices
      #######################################################
      #######################################################

      #compute gram matrices, if R is used, I need to computed, otherwise, Python returns
      #an object "kernel_" which is the kernel with the tuned hyperparameters
      #system_time_matrices captures the time for building the matrices
      system_time_matrices <- system.time({
        print("Building matrices")
        matrix_eva <- as.matrix( confounders )

        if(gpml=="R"){
          K1 <- (res.optim2_1$par[1]^2)*KGram(res.optim2_1$par,matrix_eva,kernel,degree)
          K0 <- (res.optim2_0$par[1]^2)*KGram(res.optim2_0$par,matrix_eva,kernel,degree)
        }
        if(gpml=="Python"){
          K1 <- res.optim2_1$gpr$kernel_(matrix_eva)
          K0 <- res.optim2_0$gpr$kernel_(matrix_eva)
        }

        I1 <- diag( t1 )
        I0 <- diag( t0 )

        ############################################################################
        #Compute quadratic part, Q
        #Quadratic part
        #       | ItKtIt    0     |
        # Q =   |                 |
        #       | -2KtIt     Kt   |

        IKI   <- I1%*%(K1)%*%I1 + I0%*%(K0)%*%I0
        KI    <- K1%*%I1 + K0%*%I0
        KK    <- K1 + K0
        Zero  <- 0*diag(n)

        #Compute quadratic part, Q
        Q     <- rbind( cbind( IKI, Zero ), cbind( -2*KI, KK  ) )



        #the last hyperparameter returned is sigma^2
        temp_sigma <- max(length(res.optim2_1$par))
        if(gpml=="R"){
          sigma1 <- res.optim2_1$par[temp_sigma]^2
          sigma0 <- res.optim2_0$par[temp_sigma]^2
        }
        if(gpml=="Python"){
          sigma1 <- res.optim2_1$par[temp_sigma]
          sigma0 <- res.optim2_0$par[temp_sigma]
        }

        ############################################################################
        #Compute Sigma matrix for penalization
        #           | sigmat ItIt     0   |
        # Sigma =   |                     |
        #           | 0               0   |

        tol <- 1e-08 #see comments on manuscript for tolerance in the practical guidelines
        sII <- sigma1*I1 + sigma0*I0
        sI  <- -2*(sigma1*I1 + sigma0*I0 )
        s10 <- sigma1*diag(n) + sigma0*diag(n)
        sIItemp <- tol*I1 + tol*I0

        #Sigma matrix
        Sigma <- rbind( cbind( sII, Zero ), cbind( Zero, sIItemp  ) )

        ############################################################################
        #Compute final Q, summing the initial quadratic matrix and the Sigma for
        #penalization
        #Update Q
        Q <- (1/n^2)*( Q + penal*Sigma )

      })#end system.time



      #######################################################
      #######################################################
      #Solving QP
      #######################################################
      #######################################################
      print("Solving QP")

      tol               <- 1e-08
      model             <- list()
      model$A           <- matrix(c(  (( c(t1,rep(0,n)) )/n), #sum of weights among treated
                                      (( c(t0,rep(0,n)) )/n),  #sum of weights among controls
                                      c(rep(0,n),rep(1,n))   #sum of weights V_i n/1
      ),
      nrow=3,
      byrow=T)

      model$rhs         <- c(1,1,1)                           #all sums of weights should sum up to 1
      model$modelsense  <- "min"                              #we want to minimize
      model$Q           <- Q
      model$obj         <- t(rep(0,2*n))
      model$sense       <- c("=")
      model$lb          <- c(rep(tol,n),rep(tol,n))
      model$vtypes      <- "C"

      #gurobi parameters
      params <- list(Presolve=Presolve,
                     OutputFlag=OutputFlag)

      #solve the QP
      system_time_gurobi <- system.time(res <- tryCatch(gurobi(model,params),error=function(e) NULL) )
      W <- res$x[1:n]
      V <- res$x[(n+1):(2*n)]

      if ((class(res) != "try-error") &
          (is.null(res)!=TRUE)
      ){
        result <- list(res=res,
                       WV=res$x,
                       W=res$x[1:n],
                       V=res$x[(n+1):(2*n)],
                       res.optim2_1_par = res.optim2_1$par,
                       res.optim2_0_par = res.optim2_0$par,
                       sigma0 = sigma0,
                       sigma1 = sigma1,
                       system_time_GPML0=tune_hyper$system_time_GPML0,
                       system_time_GPML1=tune_hyper$system_time_GPML1,
                       system_time_gurobi=system_time_gurobi,
                       system_time_matrices=system_time_matrices,
                       status=1)
      }


    }#end if estimand==KOWATE


    ##############################################################################################################
    ##############################################################################################################
    #*************************************************************************************************************
    #KOSATE
    #*************************************************************************************************************
    ##############################################################################################################
    ##############################################################################################################

    if(estimand=="KOSATE"){

      #######################################################
      #######################################################
      #
      #The idea of KOSATE is to follow OSATE of Crump et.al.
      #but using KOM. To do so, we need a subsample with size
      #nnt. We can obtain it by first obtaining truncated
      #weights, and then use them to get nnt. I compute them
      #following OSATE, i.e., truncating, the user only provides
      #the (upper) truncation level, e.g., 0.9
      #######################################################
      #######################################################

      #Obtain truncated indicators/weights
      fit_tr          <- glm(intervention ~ confounders, family = "binomial")
      pr1             <- fit_tr$fitted.values
      indicator_trunc <- (pr1>quantile(pr1,(1-truncation_level)))*(pr1<quantile(pr1,truncation_level))
      #sum of indicators
      ntt             <- sum(indicator_trunc)

      #######################################################
      #######################################################
      #Building matrices
      #######################################################
      #######################################################

      #compute gram matrices, if R is used, I need to computed, otherwise, Python returns
      #an object "kernel_" which is the kernel with the tuned hyperparameters
      #system_time_matrices captures the time for building the matrices
      #compute K
      system_time_matrices <- system.time({
        print("Building matrices")
        matrix_eva <- as.matrix( confounders )

        if(gpml=="R"){
          K1 <- (res.optim2_1$par[1]^2)*KGram(res.optim2_1$par,matrix_eva,kernel,degree)
          K0 <- (res.optim2_0$par[1]^2)*KGram(res.optim2_0$par,matrix_eva,kernel,degree)
        }
        if(gpml=="Python"){
          K1 <- res.optim2_1$gpr$kernel_(matrix_eva)
          K0 <- res.optim2_0$gpr$kernel_(matrix_eva)
        }


        I1 <- diag( t1 )
        I0 <- diag( t0 )

        ############################################################################
        #Compute quadratic part, Q
        #Quadratic part
        #       | ItKtIt    0     |
        # Q =   |                 |
        #       | -2KtIt     Kt   |

        IKI <- I1%*%(K1)%*%I1 + I0%*%(K0)%*%I0
        KI  <- K1%*%I1 + K0%*%I0
        KK  <- K1 + K0
        Zero <- 0*diag(n)

        Q   <- rbind( cbind( IKI, Zero ), cbind( -2*KI, KK  ) )

        #the last hyperparameter returned is sigma^2
        temp_sigma <- max(length(res.optim2_1$par))
        if(gpml=="R"){
          sigma1 <- res.optim2_1$par[temp_sigma]^2
          sigma0 <- res.optim2_0$par[temp_sigma]^2
        }
        if(gpml=="Python"){
          sigma1 <- res.optim2_1$par[temp_sigma]
          sigma0 <- res.optim2_0$par[temp_sigma]
        }

        ############################################################################
        #Compute Sigma matrix for penalization

        #           | sigmat ItIt     0   |
        # Sigma =   |                     |
        #           | 0               0   |

        tol <- 1e-08 #see comments on manuscript for tolerance in the practical guidelines
        sII <- sigma1*I1 + sigma0*I0
        sI  <- -2*(sigma1*I1 + sigma0*I0 )
        s10 <- sigma1*diag(n) + sigma0*diag(n)
        sIItemp <- tol*I1 + tol*I0

        Sigma <- rbind( cbind( sII, Zero ), cbind( Zero, sIItemp  ) )

        ############################################################################
        #Compute final Q, summing the initial quadratic matrix and the Sigma for
        #penalization
        #Update Q
        Q <- (1/n^2)*( Q + penal*Sigma )

      })#end system.time

      rm(list = c("sII","sI","s10","IKI","KI","KK","K1","K0","sIItemp"))

      #######################################################
      #######################################################
      #Solving QP
      #######################################################
      #######################################################
      print("Solving QP")

      model             <- list()
      model$A           <- matrix(c(  (( c(t1,rep(0,n)) )/n),  #sum of weights among treated
                                      (( c(t0,rep(0,n)) )/n),  #sum of weights among controls
                                      c(rep(0,n),rep(1/ntt,n)) #sum of weights V_i, ntt (sum of truncated weights)
      ), nrow=3, byrow=T)
      model$rhs         <- c(1,1,1)
      model$modelsense  <- "min"
      model$Q           <- Q
      model$obj         <- t(rep(0,2*n))
      model$sense       <- c("=")
      model$lb          <- c(rep(tol,n),rep(tol,n))
      model$vtypes      <- c(rep("C",n),rep("B",n))


      #gurobi parameters
      params <- list(Presolve=Presolve,
                     OutputFlag=OutputFlag)


      system_time_gurobi <- system.time(res <- tryCatch(gurobi(model,params),error=function(e) NULL) )
      W <- res$x[1:n]
      V <- res$x[(n+1):(2*n)]


      if ((class(res) != "try-error") &
          (is.null(res)!=TRUE)
      ){
        result <- list(res=res,
                       WV=res$x,
                       W=res$x[1:n],
                       V=res$x[(n+1):(2*n)],
                       res.optim2_1_par = res.optim2_1$par,
                       res.optim2_0_par = res.optim2_0$par,
                       sigma0 = sigma0,
                       sigma1 = sigma1,
                       system_time_GPML0=tune_hyper$system_time_GPML0,
                       system_time_GPML1=tune_hyper$system_time_GPML1,
                       system_time_gurobi=system_time_gurobi,
                       system_time_matrices=system_time_matrices,
                       status=1)
      }#end if error result


    }#end if estimand==KOSATE


    ##############################################################################################################
    ##############################################################################################################
    #*************************************************************************************************************
    #SATE
    #*************************************************************************************************************
    ##############################################################################################################
    ##############################################################################################################

    if(estimand=="SATE"){
      #compute gram matrices, if R is used, I need to computed, otherwise, Python returns
      #an object "kernel_" which is the kernel with the tuned hyperparameters
      #system_time_matrices captures the time for building the matrices
      #compute K
      system_time_matrices <- system.time({
        matrix_eva <- as.matrix( confounders )

        if(gpml=="R"){
          K1 <- (res.optim2_1$par[1]^2)*KGram(res.optim2_1$par,matrix_eva,kernel,degree)
          K0 <- (res.optim2_0$par[1]^2)*KGram(res.optim2_0$par,matrix_eva,kernel,degree)
        }
        if(gpml=="Python"){
          K1 <- res.optim2_1$gpr$kernel_(matrix_eva)
          K0 <- res.optim2_0$gpr$kernel_(matrix_eva)
        }

        print("Building matrices")
        # Recall that V <- rep(1/n,n) target SATE.
        #A user may want to use another set of pre-computed weights, e.g., overlap weights/
        #truncated weights and optain the set of weights that minimize the CMSE of the
        #weighted estimatort for GATE in that specific population, i.e., the one defined
        #by the pre-computed weights. If nothing is provided SATE with 1/n is computed
        V <- weights/n

        ############################################################################
        #Compute quadratic part, Q
        #Quadratic part
        I1    <- diag( t1 )
        I0    <- diag( t0 )
        I1KI1 <- I1%*%K1%*%I1
        I0KI0 <- I0%*%K0%*%I0

        ############################################################################
        #Compute linear part, c
        #Linear part
        # V^T K1 I1 + V^T K0 I0
        #K1 I1, K0 I0
        KI1 <- I1%*%K1
        KI0 <- I0%*%K0

        VKI1 <- t(V)%*%KI1
        VKI0 <- t(V)%*%KI0



        #the last hyperparameter returned is sigma^2
        temp_sigma <- max(length(res.optim2_1$par))
        if(gpml=="R"){
          sigma1 <- res.optim2_1$par[temp_sigma]^2
          sigma0 <- res.optim2_0$par[temp_sigma]^2
        }
        if(gpml=="Python"){
          sigma1 <- res.optim2_1$par[temp_sigma]
          sigma0 <- res.optim2_0$par[temp_sigma]
        }

        ############################################################################
        #Compute Sigma matrix for penalization
        Sigma <- sigma1*diag(t1) + sigma0*diag(t0)

        ############################################################################
        #Compute final Q and c, summing the initial quadratic matrix and the Sigma for
        #penalization
        #Update Q
        Q <- (1/n^2)*( I1KI1 + I0KI0 + penal*Sigma )

        #Update c
        #Gurobi:
        c <- -2*(1/n^2)*(VKI1 + VKI0)

        rm(list = c("VKI1","VKI0"))

      })#end system.time


      #######################################################
      #######################################################
      #Solving QP
      #######################################################
      #######################################################

      print("Solving QP")

      tol               <- 1e-08
      model             <- list()
      model$A           <- matrix(c( t1/n ,t0/n), nrow=2, byrow=T) #sum of weights treated/controls to 1
      model$rhs         <- c(1,1)
      model$modelsense  <- "min"
      model$Q           <- Q
      model$obj         <- c
      model$sense       <- c("=")
      model$lb          <- rep(tol,n)
      model$vtypes      <- "C"


      #gurobi parameters
      params <- list(Presolve=Presolve,
                     OutputFlag=OutputFlag)

      system_time_gurobi <- system.time(res <- tryCatch(gurobi(model,params),error=function(e) NULL))

      if ( (class(res) != "try-error") &
           (is.null(res)!=TRUE)
      ){
        result <- list(res=res,
                       W=res$x,
                       res.optim2_1_par = res.optim2_1$par,
                       res.optim2_0_par = res.optim2_0$par,
                       sigma0 = sigma0,
                       sigma1 = sigma1,
                       system_time_GPML0=tune_hyper$system_time_GPML0,
                       system_time_GPML1=tune_hyper$system_time_GPML1,
                       system_time_gurobi=system_time_gurobi,
                       system_time_matrices=system_time_matrices,
                       status=1)
      }#end if error result

    }#end if estimand==SATE


    ##############################################################################################################
    ##############################################################################################################
    #*************************************************************************************************************
    #TATE
    #*************************************************************************************************************
    ##############################################################################################################
    ##############################################################################################################

    if(estimand=="TATE"){
      #compute gram matrices, if R is used, I need to computed, otherwise, Python returns
      #an object "kernel_" which is the kernel with the tuned hyperparameters
      #system_time_matrices captures the time for building the matrices
      #compute K
      system_time_matrices <- system.time({

        matrix_eva <- as.matrix( confounders )

        if(gpml=="R"){
          K1 <- (res.optim2_1$par[1]^2)*KGram(res.optim2_1$par,matrix_eva,kernel,degree)
          K0 <- (res.optim2_0$par[1]^2)*KGram(res.optim2_0$par,matrix_eva,kernel,degree)
        }
        if(gpml=="Python"){
          K1 <- res.optim2_1$gpr$kernel_(matrix_eva)
          K0 <- res.optim2_0$gpr$kernel_(matrix_eva)
        }

        #U is the indicator of being in the target population
        #S is the indicator of being in the sample population
        u         <- 1-sample_population
        n_target  <- sum(u)
        n_sample  <- dim(confounders)[1]-n_target
        n         <- (n_target+n_sample)

        S <- diag(sample_population)
        U <- diag(u)
        V <- rep(n/n_target,n)

        I1 <- diag( c(t1,rep(1,n_target)) )
        I0 <- diag( c(t0,rep(1,n_target)) )

        ############################################################################
        #Compute quadratic part, Q
        #Quadratic part
        # (S) (K1 + K0) (S)
        SI1     <- S%*%I1
        SI0     <- S%*%I0
        SK1K0S  <- SI1%*%(K1)%*%SI1 + SI0%*%(K0)%*%SI0

        ############################################################################
        #Compute linear part, c
        #Linear part
        #V^T K1 I1 + V^T K0 I0
        #K1 I1, K0 I0
        SK1 <- K1%*%S%*%I1
        SK0 <- K0%*%S%*%I0

        # V^T K1 I1, V^T K0 I0
        VSK1 <- t(V)%*%SK1
        VSK0 <- t(V)%*%SK0



        #the last hyperparameter returned is sigma^2
        temp_sigma <- max(length(res.optim2_1$par))
        if(gpml=="R"){
          sigma1 <- res.optim2_1$par[temp_sigma]^2
          sigma0 <- res.optim2_0$par[temp_sigma]^2
        }
        if(gpml=="Python"){
          sigma1 <- res.optim2_1$par[temp_sigma]
          sigma0 <- res.optim2_0$par[temp_sigma]
        }

        ############################################################################
        #Compute Sigma matrix for penalization
        Sigma <- sigma1*diag( c(t1,rep(0,n_target)) ) + sigma0*diag( c(t0,rep(0,n_target))  )


        ############################################################################
        #Compute final Q and c, summing the initial quadratic matrix and the Sigma for
        #penalization
        #Update Q
        penal <- 1
        Q <- (1/n^2)*( SK1K0S + penal*Sigma )

        #Update c
        #Gurobi:
        c <- -2*(1/n^2)*(VSK1 + VSK0)

      })#end system.time

      #######################################################
      #######################################################
      #Solving QP
      #######################################################
      #######################################################
      print("Solving QP")
      tol               <- 1e-08
      model             <- list()
      model$A           <- matrix(c(( c(t1,rep(0,n_target))/n_sample),
                                    ( c(t0,rep(0,n_target))/n_sample)),
                                  nrow=2,
                                  byrow=T)
      model$rhs         <- c(1,1)
      model$modelsense  <- "min"
      model$Q           <- Q
      model$obj         <- c
      model$sense       <- c("=")
      model$lb          <- rep(tol,n)
      model$vtypes      <- "C"


      #gurobi parameters
      params <- list(Presolve=Presolve,
                     OutputFlag=OutputFlag)

      system_time_gurobi <- system.time(res <- tryCatch(gurobi(model,params),error=function(e) NULL))

      if ( (class(res) != "try-error") &
           (is.null(res)!=TRUE)
      ){
        result <- list(res=res,W=res$x,
                       res.optim2_1_par = res.optim2_1$par,
                       res.optim2_0_par = res.optim2_0$par,
                       sigma0 = sigma0,
                       sigma1 = sigma1,
                       system_time_GPML0=tune_hyper$system_time_GPML0,
                       system_time_GPML1=tune_hyper$system_time_GPML1,
                       system_time_gurobi=system_time_gurobi,
                       system_time_matrices=system_time_matrices,
                       status=1)
      }#end if results not error


    }#end if estimand==TATE

  }#end if ERROR
  if ( (class(res.optim2_0) == "try-error") |
       (class(res.optim2_1) == "try-error") |
       (is.null(res)==TRUE) | (is.null(res.optim2_0)==TRUE) |
       (is.null(res.optim2_1)==TRUE) ){
    result <- list(res=NA,
                   WV=rep(NA,n),
                   W=rep(NA,n),
                   V=rep(NA,n),
                   res.optim2_1_par = rep(NA,4),
                   res.optim2_0_par = rep(NA,4),
                   sigma0 = NA,
                   sigma1 = NA,
                   system_time_GPML0=tune_hyper$system_time_GPML0,
                   system_time_GPML1=tune_hyper$system_time_GPML1,
                   system_time_gurobi=NA,
                   system_time_matrices=NA,
                   status=0)
    print("------------------ ERROR ------------------")
  }
  return(result)

}

