import numpy as np

# nonlinearity sigmoid function
def sigmoid(input):
    return 1 / (1 + np.exp(-input))

# used to compute the gradients in backprop
def derivative_sigmoid( input):
    return sigmoid(input)*(1 - sigmoid(input))

# tanh activation function, used in the LSTM cells
def tanh(input):
    return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

# derivative for computing gradients
def derivative_tanh(input):
    return 1 - (tanh(input) * tanh(input))

def leaky_ReLu(input):
    np.maximum(input, 0.01*input, input)

def derivate_LReLu(input):
    np.maximum(input, .01, input)