#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:44:53 2018

@author: chrismiller
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import beta

def plot_laplace(a, b, N, y):
    mu = (y + a - 1.0) / (a + N + b - 2.0)
    mu2 = np.power(mu, 2)
    q2 = np.power((mu - 1), 2)
    
    sigma2 = (mu2 * q2) / (q2 * (y + a - 1) + mu2 * (N - y + b - 1))
    
    print("mean: ", mu)
    print("variance: ", sigma2)
    
    x = np.arange(0, 1, 0.01)
    ybeta = beta.pdf(x, a + y, b + N - y)
    ynorm = norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))
    
    plt.figure()
    plt.plot(x, ybeta, label='True Beta')
    plt.plot(x, ynorm, label='Laplace Estimation')
    plt.legend()

print("--------part 1--------")    
plot_laplace(a=5, b=5, N=20, y=10)

print("\n--------part 2--------")
plot_laplace(a=3, b=15, N=10, y=3)

print("\n--------part 3--------")
plot_laplace(a=1, b=30, N=10, y=3)
