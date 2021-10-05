#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:57:16 2020

@author: macbookair
"""

import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from time import time
from numpy.random import normal
from numpy.random import multivariate_normal
from scipy.stats import norm

data =[]

# Number of gaussians
Ng = 1

# Dimention
D = 2

# length 
L = 10

Center = np.zeros((Ng, D))
Center[0] = 2*np.ones(D)
#Center[1] = 8*np.ones(D)

# Standard deviation 
#Sigma = np.array([0.5, 0.5])
Sigma = np.array([0.5])

# Number of generated data
M = np.zeros(Ng, dtype = int)

nb = 10000
C = np.sqrt((1/nb)*np.sum((Sigma)**2)) 

# ratio signal over noise
#R = 0.01 # PROVISOIRE

# Generating data
for i in range(Ng) :
    #M[i] = int( (Sigma[i] / R)**2 )
    M[i] = int( (Sigma[i] / C)**2 )
    #print(Center[i])
    
    for k in range(M[i]) :
        X = normal( Center[i], Sigma[i]**2)
        X = np.maximum(X, 0)
        X = np.minimum(X, L)
        
        data.append(X)

data = np.array(data)        

# Centering data
mean = np.sum(data, axis=0) / M.sum()
data = data - mean

np.random.shuffle(data)

print()
print(M)
print(data.shape)

"""
U, s, Vs = np.linalg.svd(data, full_matrices=True)

P = np.dot(data, Vs.T)

print(P.shape)
print(s.shape)

N = np.sum(M)
"""

def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca, eigen_vecs

def reduc(x, V):
    return np.dot(x, V)[1:3]

X_pca, V = pca(data)

#X = X_pca[:, 0]
#Y = X_pca[:, 1]

X = data[:, 0]
Y = data[:, 1]
plt.title("Selon les premières coordonées")
plt.plot(X, Y, 'x')
plt.show()

X = X_pca[:, 0]
Y = X_pca[:, 1]
plt.title("Selon les composantes principales")
plt.plot(X, Y, 'x')
plt.show()

np.save('V.npy', V)
np.save('data.npy', data)

    
    