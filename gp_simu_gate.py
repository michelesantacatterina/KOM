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
# the potential outcome Yi(a) from f(Xi) with Gaussian noise of variance sigma2.  We then maximize the marginal likelihood
# of seeing the data with respect to the hyperparameters.
#
# This is the python file for GaussianProcessRegressor
#*****************************************************************************************************************************
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, Product, Sum, ConstantKernel, RBF, Matern, ExpSineSquared
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import numpy as np


##############################################################################################################################
#
# gp: function for gaussian process regressor
#
# x: confounders
# y: outcome
# degree1: if you have a combination of kernels (like the product of 2 kernels) this is the degree of the first kernel
# degree2: degree of the second kernel
# k1: first kernel
# k2: second kernel
# operator: single, sum or product of kernels.
#
# return: the object gpr from GaussianProcessRegressor and the Gram matrix based on the tuned hyperparameters.
##############################################################################################################################


def gp(x,y,degree1,degree2,k1,k2,operator):
  ################################## ################################## ##################################  
  #product operator
  if operator == "prod":
    #polynomial * polynomial
    if k1 == "poly" and k2 == "poly":
      k     =  ConstantKernel()*( (DotProduct() ** degree1)*(DotProduct() ** degree2) )  + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)

    #rbf * rbf
    if k1 == "rbf" and k2 == "rbf":
      k     = ConstantKernel()*( (RBF())*(RBF()) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)

    #polynomial * rbf
    if (k1 == "poly" and k2 == "rbf") or (k1 == "rbf" and k2 == "poly"):
      k     = ConstantKernel()*( (DotProduct() ** degree1)*(RBF()) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)

    #matern nu=3/2 * matern nu=3/2
    if k1 == "matern1.5" and k2 == "matern1.5":
      k     = ConstantKernel()*( ( Matern(nu=1.5) )*( Matern(nu=1.5)  ) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
      
    #matern nu=5/2 * matern nu=5/2      
    if k1 == "matern2.5" and k2 == "matern2.5":
      k     = ConstantKernel()*( ( Matern(nu=2.5) )*( Matern(nu=2.5)  ) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
      
    #poly * matern nu=3/2        
    if (k1 == "poly" and k2 == "matern1.5") or (k1 == "matern1.5" and k2 == "poly"):
      k     = ConstantKernel()*( ( DotProduct() ** degree1 )*( Matern(nu=1.5)  ) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y) 
      
    #poly * matern nu=5/2    
    if (k1 == "poly" and k2 == "matern2.5") or (k1 == "matern2.5" and k2 == "poly"):
      k     = ConstantKernel()*( ( DotProduct() ** degree1 )*( Matern(nu=2.5)  ) ) + WhiteKernel()
      gpr   = GaussianProcessRegressor(kernel=k).fit(x,y) 
    
      
  ################################## ################################## ##################################    
  #sum operator    
  if operator == "sum":
      #polynomial + polynomial
      if k1 == "poly" and k2 == "poly":
        k     =  ConstantKernel()*( (DotProduct() ** degree1) + (DotProduct() ** degree2) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)

      #rbf + rbf
      if k1 == "rbf" and k2 == "rbf":
        k     = ConstantKernel()*( (RBF()) + (RBF()) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #polynomial + rbf
      if (k1 == "poly" and k2 == "rbf") or (k1 == "rbf" and k2 == "poly"):
        k     = ConstantKernel()*( (DotProduct() ** degree1) + (RBF()) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #matern nu=3/2 + matern nu=3/2
      if k1 == "matern1.5" and k2 == "matern1.5":
        k     = ConstantKernel()*( ( Matern(nu=1.5) ) + ( Matern(nu=1.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #matern nu=5/2 + matern nu=5/2      
      if k1 == "matern2.5" and k2 == "matern2.5":
        k     = ConstantKernel()*( ( Matern(nu=2.5) ) + ( Matern(nu=2.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #poly + matern nu=3/2        
      if (k1 == "poly" and k2 == "matern1.5") or (k1 == "matern1.5" and k2 == "poly"):
        k     = ConstantKernel()*( ( DotProduct() ** degree1 ) + ( Matern(nu=1.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y) 
        
      #poly + matern nu=5/2  
      if (k1 == "poly" and k2 == "matern2.5") or (k1 == "matern2.5" and k2 == "poly"):
        k     = ConstantKernel()*( ( DotProduct() ** degree1 ) + ( Matern(nu=2.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y) 

  ################################## ################################## ##################################  
  #single operator
  if operator == "single":
      #polynomial
      if k1 == "poly":
        k     = ConstantKernel()*( ( DotProduct()**degree1  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #rbf
      if k1 == "rbf":
        k     = ConstantKernel()*( ( RBF()  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)
        
      #matern1.5  
      if k1 == "matern1.5":
        k     = ConstantKernel()*( ( Matern(nu=1.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y)   
      
      #matern2.5    
      if k1 == "matern2.5":
        k     = ConstantKernel()*( ( Matern(nu=2.5)  ) ) + WhiteKernel()
        gpr   = GaussianProcessRegressor(kernel=k).fit(x,y) 

  #get gram matrix
  GramM = gpr.kernel_(x)


  #return gpr from GaussianProcessRegressor and Gram matrix
  return {'gpr': gpr, 'GramMatrix': GramM}
  
  
  
  
  
  
  
  
  
