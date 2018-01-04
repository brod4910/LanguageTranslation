import numpy as np

# Implementation of the cross entropy loss function.
# Used for calculating our loss after we have finished our forward pass
def cross_entropy_loss(output, labels):
	y_hat = labels.shape[0]
	probabilities = softmax(output)
	n_log_likelihood =  -np.log(probabilities[range(m),labels])
	loss = np.sum(n_log_likelihood) / y_hat
	return loss

# Derivatitve of the cross entropy loss functions.
# Used to compute the gradients in backprop
def deriv_cross_entropy(output, labels):
	y_hat = labels.shape[0]
	gradients = softmax(output)
	gradients[range(y_hat),labels] -= 1
	gradients = gradients/y_hat
	return gradients

# Implementation of the softmax loss function.
# take the max of the input to get the negative numbers be zero
# to prevent NaN overflowing in our computations
def softmax(input):
    input -= np.max(input)
    return np.exp(input) / np.sum(np.exp(input))