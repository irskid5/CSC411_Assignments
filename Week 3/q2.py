# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    # Create the A array
    A = np.zeros((x_train.shape[0], x_train.shape[0]))

    # Create dist array
    dist = l2(test_datum.reshape([1, d]), x_train)

    # Fill A matrix
    num = -1/(2*tau**2) * dist.T
    A = np.exp(num) / np.exp(logsumexp(num))
    A = np.diag(A[:, 0])

    # Calculate w_star
    w_star = np.linalg.solve( ( np.dot( x_train.T, np.dot(A, x_train) ) + lam*np.identity(d) ), np.dot( x_train.T, np.dot(A, y_train) ) )

    # Return y_hat as desired
    return np.dot(test_datum.T, w_star)

def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''

    # Divide data
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=val_frac)

    # Compute the losses
    l_train = []
    l_val = []
    for t in taus:
        loss_train = 0
        loss_val = 0
        for x_i, y in zip(x_train, y_train):
            y_hat = LRLS(x_i.T, x_train, y_train, t)
            loss_train += 1/2 * (y_hat - y) ** 2

        for x_i, y in zip(x_val, y_val):
            y_hat = LRLS(x_i.T, x_train, y_train, t)
            loss_val += 1/2 * (y_hat - y) ** 2
        l_train.append(loss_train/len(y_train))
        l_val.append(loss_val/len(y_val))

    return l_train, l_val


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
    plt.xlabel("Taus")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss in Locally Reweighted Least Square")
    plt.legend(["Training", "Validation"])
    plt.show()
