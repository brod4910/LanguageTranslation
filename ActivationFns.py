class ActivationFns:
    # nonlinearity sigmoid function
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    # used to compute the gradients in backprop
    def derivative_sigmoid(self, input):
        return self.sigmoid(input)*(1 - self.sigmoid(input))

    # tanh activation function, used in the LSTM cells
    def tanh(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    # derivative for computing gradients
    def derivative_tanh(self, input):
        return 1 - (self.tanh(input) * self.tanh(input))

    def leaky_ReLu(self, input):
        np.maximum(input, 0.01*input, input)

    def derivate_LReLu(self, input):
        np.maximum(input, .01, input)