'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))

    # Compute means
    train_data_with_labels = np.c_[train_data, train_labels]

    for i in range(10):
        digit_matrix_with_label = train_data_with_labels[train_data_with_labels[:,-1] == i]
        digit_matrix_no_label = digit_matrix_with_label[:, 0:-1]
        means[i] = np.sum(digit_matrix_no_label, axis=0) / digit_matrix_no_label.shape[0]

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances

    train_data_with_labels = np.c_[train_data, train_labels]
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        digit_matrix_with_label = train_data_with_labels[train_data_with_labels[:,-1] == i]
        digit_matrix_no_label = digit_matrix_with_label[:, 0:-1]
        sigma = digit_matrix_no_label - means[i]
        covariances[i] = np.dot(sigma.T, sigma) / sigma.shape[0] + 0.01 * np.eye(means.shape[1])

    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''

    generative_likelihood = np.zeros((digits.shape[0], 10))

    for i in range(means.shape[0]):
        covariances_inverse = np.linalg.inv(covariances[i])
        covariances_det = np.linalg.det(covariances[i])
        sigma = digits - means[i]
        inner_dot = np.dot(sigma, covariances_inverse)
        ex_value = np.dot(inner_dot, sigma.T)
        p = ((2 * np.pi) ** (-means.shape[1]/2)) * (covariances_det ** (-1/2)) * np.exp((-1/2) * ex_value)
        p_log = np.log(p)
        column = np.diag(p_log)
        generative_likelihood[:,i] = column

    return generative_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    gen_lik = generative_likelihood(digits, means, covariances)
    prior = np.ones((gen_lik.shape[0], 1)) * 1/10

    return gen_lik + np.log(prior)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    avg_cond_lik = np.zeros((10, 10))
    cond_lik = conditional_likelihood(digits, means, covariances)
    cond_lik_with_labels = np.c_[cond_lik, labels]
    for i in range(10):
        cond_lik_matrix_with_label = cond_lik_with_labels[cond_lik_with_labels[:,-1] == i]
        cond_lik_matrix_no_label = cond_lik_matrix_with_label[:, 0:-1]
        avg_cond_lik[i] = np.sum(cond_lik_matrix_no_label, axis=0) / cond_lik_matrix_no_label.shape[0]

    avg = 0
    for i in range(0, digits.shape[0]):
        k = labels.astype(int)[i]
        avg += cond_lik[i, k]

    avg /= digits.shape[0]

    print("Average Conditional Likelihood (as a number): %f" % avg)

    return avg_cond_lik

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''

    cond_likelihood = conditional_likelihood(digits, means, covariances)

    return np.argmax(cond_likelihood, axis=1)


# noinspection SpellCheckingInspection
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

    # 1. a)
    avg_cond_lik_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_lik_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("1. a)")
    print("Average Conditional Likelihood on Train set: \n", avg_cond_lik_train)
    print("Average Conditional Likelihood on Test set: \n", avg_cond_lik_test)

    # 1. b)
    train_prediction = classify_data(train_data, means, covariances)
    train_labels = train_labels.astype(int)
    correct_matrix = np.where(train_prediction == train_labels)
    train_accuracy = (correct_matrix[0].shape[0] / train_labels.shape[0]) * 100

    test_predication = classify_data(test_data, means, covariances)
    test_labels = test_labels.astype(int)
    correct_matrix = np.where(test_predication == test_labels)
    test_accuracy = (correct_matrix[0].shape[0] / test_labels.shape[0]) * 100

    print("\n1. b)")
    print("Train accuracy (in percent): %f" % train_accuracy)
    print("Test accuracy (in percent): %f" % test_accuracy)

    # 1. c)
    for i, covariance in enumerate(covariances):
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        leading_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        plt.subplot(2, 5, i + 1)
        plt.imshow(leading_eigenvector.reshape(8, 8))
        plt.xlabel("Digit=%d" % i)

    plt.show()


if __name__ == '__main__':
    main()
