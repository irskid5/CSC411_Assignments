import numpy as np
def gradient_descent(X, y, lr, num_iter, delta):
    T = np.copy(y)
    Y = np.zeros(len(T))
    W, B = np.zeros(len(T)), np.zeros(len(T))
    for i in range (num_iter):
        Y = np.dot(X, W) + B
        A = Y - T
        W -= (lr/len(A)) * gradientW(X, A, delta)
        B -= (lr/len(B)) * gradientB(X, A, delta)

    return W, B

def gradientW(X, A, delta):

    ACond1 = np.copy(A)
    ACond2 = delta*np.sign(A)
    HDecision = np.where(abs(A) <= delta*np.ones(len(A)), ACond1, ACond2)
    return np.dot(HDecision.T, X).T

def gradientB(X, A, delta):

    ACond1 = np.copy(A)
    ACond2 = delta*np.sign(A)
    HDecision = np.where(abs(A) <= delta*np.ones(len(A)), ACond1, ACond2)
    return np.dot(HDecision.T, np.ones(X.shape)).T
