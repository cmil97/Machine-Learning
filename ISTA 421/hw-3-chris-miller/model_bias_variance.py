#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:55:47 2018

@author: chrismiller
"""

import numpy
import matplotlib.pyplot as plt

# set to True in order to automatically save the generated plots
SAVE_FIGURES = True

# change this to where you'd like the figures saved
# (relative to your python current working directory)
FIGURE_PATH = '/Users/chrismiller/Downloads/ista421ML-f2018-hw3-release/code'

def true_function(x):
    """$t = 5x+x^2-0.5x^3$"""
    return (5 * x) + x**2 - (0.5 * x**3)


def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    """ Sample data from the true function.
        N: Number of samples
        Returns a noisy sample t_sample from the function
        and the true function t. """
    x = numpy.random.uniform(xmin, xmax, N)
    t = true_function(x)
    # add standard normal noise using numpy.random.randn
    # (standard normal is a Gaussian N(0, 1.0)  (i.e., mean 0, variance 1),
    #  so multiplying by numpy.sqrt(noise_var) make it N(0,standard_deviation))
    t = t + numpy.random.randn(x.shape[0])*numpy.sqrt(noise_var)
    return x, t

def calculate_cov_w(X, w, t):
    """
    Calculates the covariance of w
    :param X: Design matrix: matrix of N observations
    :param w: vector of parameters
    :param t: vector of N target responses
    :return: the matrix covariance of w
    """
    ### Insert your code here to calculate the estimated covariance of w ###
    covw = None
    
    N = X.shape[0]
    variance = (1/N)*(numpy.dot(t.T, t) - numpy.dot(numpy.dot(t, X), w))
    
    covw = variance * numpy.linalg.inv(numpy.dot(X.T, X))
    
    return covw

# 20 data sets (models)
# 25 samples (data points)
# range -4 to 5, noise = 6 (sample_from_function)

#for each model (1, 3, 5, 9)
#plot each model (blue)
#plot true function (red, linewidth=3)
    
xmin = -4.
xmax = 5.
noise_var = 6

# sample 25 points from function
x, t = sample_from_function(25, noise_var, xmin, xmax)

# Plot just the sampled data
plt.figure(0)
plt.scatter(numpy.asarray(x), numpy.asarray(t), color='k', edgecolor='k')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sampled data from {0}, $x \in [{1},{2}]$'
          .format(true_function.__doc__, xmin, xmax))
plt.pause(.1)  # required on some systems so that rendering can happen

if SAVE_FIGURES:
    plt.savefig(FIGURE_PATH + 'data.pdf', fmt='pdf')

# Fit models of various orders
orders = [1, 3, 5, 9]

# Make a set of 100 evenly-spaced x values between xmin and xmax
testx = numpy.linspace(xmin, xmax, 100)

# Generate plots of functions whose parameters are sampled based on cov(\hat{w})
num_function_samples = 20
for i in orders:
    # create input representation for given model polynomial order
    X = numpy.zeros(shape=(x.shape[0], i+1))
    testX = numpy.zeros(shape=(testx.shape[0], i+1))
    for k in range(i + 1):
        X[:, k] = numpy.power(x, k)
        testX[:, k] = numpy.power(testx, k)

    # fit model parameters
    w = numpy.dot(numpy.linalg.inv(numpy.dot(X.T, X)), numpy.dot(X.T, t))
    
    # Sample functions with parameters w sampled from a Gaussian with
    # $\mu = \hat{\mathbf{w}}$
    # $\Sigma = cov(w)$

    # determine cov(w)
    covw = calculate_cov_w(X, w, t)  # calculate the covariance of w

    # The following samples num_function_samples of w from
    # multivariate Gaussian (normal) with covaraince covw
    wsamp = numpy.random.multivariate_normal(w, covw, num_function_samples)

    # Calculate means for each function
    prediction_t = numpy.dot(testX, wsamp.T)
    
    # Plot the data and functions
    plt.figure()
    plt.scatter(x, t, color='k', edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.plot(testx, prediction_t, color='b')
    plt.plot(testx, true_function(testx), color='r', linewidth=3)
    
    # find reasonable ylim bounds
    min_model = min(prediction_t.flatten())
    max_model = max(prediction_t.flatten())
    min_testvar = min(min(t), min_model)
    max_testvar = max(max(t), max_model)
    plt.ylim(min_testvar, max_testvar)  # (-400,400)
    
    ti = 'Plot of {0} functions where parameters '\
         .format(num_function_samples, i) + \
         r'$\widehat{\bf w}$ were sampled from' + '\n' + r'cov($\bf w$)' + \
         ' of model with polynomial order {1}' \
         .format(num_function_samples, i)
    plt.title(ti)
    plt.pause(.1)  # required on some systems so that rendering can happen
    
    if SAVE_FIGURES:
        filename = 'sampled-fns-{0}.pdf'.format(i)
        plt.savefig(FIGURE_PATH + filename, fmt='pdf')

plt.show()