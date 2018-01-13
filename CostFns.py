import numpy as np

# Implementation of the cross entropy loss function.
# Used for calculating our loss after we have finished our forward pass
def cross_entropy_loss(output, labels):
	label_size = labels.shape[1]
	label_indicies = np.argmax(labels, axis=1)
	probabilities = softmax(output)
	n_log_likelihood =  -np.log(probabilities[np.arange(len(probabilities)), label_indicies])
	loss = np.sum(n_log_likelihood) / label_size
	return loss

# Derivatitve of the cross entropy loss functions.
# Used to compute the gradients in backprop
def deriv_cross_entropy(output, labels):
	label_size = labels.shape[0]
	gradients = softmax(output)
	gradients[range(label_size),labels] -= 1
	gradients = gradients/label_size
	return gradients

# Implementation of the softmax loss function.
# take the max of the input to get the negative numbers to be zero
# to prevent NaN overflowing in our computations
def softmax(input):
    input -= np.max(input)
    return np.exp(input) / np.sum(np.exp(input))