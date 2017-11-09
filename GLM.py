# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 06:23:25 2017

@author: Y9CK3
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import statsmodels.api as sm
%matplotlib inline

# Load iris dataset
iris = datasets.load_iris()
print(iris.data.shape)
print(iris.data[0:10,:])
y = iris.data[:,3]
X = iris.data[:,0:3]

# GAMMA 
gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()
gamma_results.params
gamma_results.scale
gamma_results.deviance
gamma_results.pearson_chi2
gamma_results.llf
print(gamma_results.summary())

# INVERSE GAUSSIAN
inverse_gaussion_model = sm.GLM(y, X, family=sm.families.InverseGaussian())
inverse_gaussion_results = gamma_model.fit()
inverse_gaussion_results.params
inverse_gaussion_results.scale
inverse_gaussion_results.deviance
inverse_gaussion_results.pearson_chi2
inverse_gaussion_results.llf
print(inverse_gaussion_results.summary())


# POISSON
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
poisson_results.params
poisson_results.scale
poisson_results.deviance
poisson_results.pearson_chi2
poisson_results.llf
print(poisson_results.summary())

# BINOMIAL 
binomial_model = sm.GLM(y, X, family=sm.families.Binomial())
binomial_results = binomial_model.fit()
binomial_results.params
binomial_results.scale
binomial_results.deviance
binomial_results.pearson_chi2
binomial_results.llf
print(binomial_results.summary())


# GAUSSIAN
gaussian_model = sm.GLM(y, X, family=sm.families.Gaussian())
gaussian_results = gaussian_model.fit()
gaussian_results.params
gaussian_results.scale
gaussian_results.deviance
gaussian_results.pearson_chi2
gaussian_results.llf
print(gaussian_results.summary())

# NEGATIVE BINOMIAL
negative_binomial_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
negative_binomial_results = negative_binomial_model.fit()
negative_binomial_results.params
negative_binomial_results.scale
negative_binomial_results.deviance
negative_binomial_results.pearson_chi2
negative_binomial_results.llf
print(negative_binomial_results.summary())



# Load boston housing dataset
boston = datasets.load_boston()
print(boston.data.shape)
print(boston.data[0:10,:])
y = boston.data[:,12]
X = boston.data[:,0:12]

# GAMMA 
gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()
gamma_results.params
gamma_results.scale
gamma_results.deviance
gamma_results.pearson_chi2
gamma_results.llf
print(gamma_results.summary())

# INVERSE GAUSSIAN
inverse_gaussion_model = sm.GLM(y, X, family=sm.families.InverseGaussian())
inverse_gaussion_results = gamma_model.fit()
inverse_gaussion_results.params
inverse_gaussion_results.scale
inverse_gaussion_results.deviance
inverse_gaussion_results.pearson_chi2
inverse_gaussion_results.llf
print(inverse_gaussion_results.summary())


# POISSON
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
poisson_results.params
poisson_results.scale
poisson_results.deviance
poisson_results.pearson_chi2
poisson_results.llf
print(poisson_results.summary())

# BINOMIAL 
binomial_model = sm.GLM(y, X, family=sm.families.Binomial())
binomial_results = binomial_model.fit()
binomial_results.params
binomial_results.scale
binomial_results.deviance
binomial_results.pearson_chi2
binomial_results.llf
print(binomial_results.summary())


# GAUSSIAN
gaussian_model = sm.GLM(y, X, family=sm.families.Gaussian())
gaussian_results = gaussian_model.fit()
gaussian_results.params
gaussian_results.scale
gaussian_results.deviance
gaussian_results.pearson_chi2
gaussian_results.llf
print(gaussian_results.summary())

# NEGATIVE BINOMIAL
negative_binomial_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
negative_binomial_results = negative_binomial_model.fit()
negative_binomial_results.params
negative_binomial_results.scale
negative_binomial_results.deviance
negative_binomial_results.pearson_chi2
negative_binomial_results.llf
print(negative_binomial_results.summary())




# Load boston housing dataset
diabetes = datasets.load_diabetes()
print(diabetes.data.shape)
print(diabetes.data[0:10,:])
y = diabetes.data[:,9]
X = diabetes.data[:,0:9]

# GAMMA 
gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()
gamma_results.params
gamma_results.scale
gamma_results.deviance
gamma_results.pearson_chi2
gamma_results.llf
print(gamma_results.summary())

# INVERSE GAUSSIAN
inverse_gaussion_model = sm.GLM(y, X, family=sm.families.InverseGaussian())
inverse_gaussion_results = gamma_model.fit()
inverse_gaussion_results.params
inverse_gaussion_results.scale
inverse_gaussion_results.deviance
inverse_gaussion_results.pearson_chi2
inverse_gaussion_results.llf
print(inverse_gaussion_results.summary())


# POISSON
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
poisson_results.params
poisson_results.scale
poisson_results.deviance
poisson_results.pearson_chi2
poisson_results.llf
print(poisson_results.summary())

# BINOMIAL 
binomial_model = sm.GLM(y, X, family=sm.families.Binomial())
binomial_results = binomial_model.fit()
binomial_results.params
binomial_results.scale
binomial_results.deviance
binomial_results.pearson_chi2
binomial_results.llf
print(binomial_results.summary())


# GAUSSIAN
gaussian_model = sm.GLM(y, X, family=sm.families.Gaussian())
gaussian_results = gaussian_model.fit()
gaussian_results.params
gaussian_results.scale
gaussian_results.deviance
gaussian_results.pearson_chi2
gaussian_results.llf
print(gaussian_results.summary())

# NEGATIVE BINOMIAL
negative_binomial_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
negative_binomial_results = negative_binomial_model.fit()
negative_binomial_results.params
negative_binomial_results.scale
negative_binomial_results.deviance
negative_binomial_results.pearson_chi2
negative_binomial_results.llf
print(negative_binomial_results.summary())
























