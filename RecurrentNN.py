import numpy as np
import os
import LSTMcell
import ActivationFns as af
import CostFns as cf

"""

"""
class RecurrentNeuralNetwork:
    # args: input_size = length of longest sentence in either language, 
    # output_size = the same # as input_size. This is only in for readability, 
    # input_vocab_size = vocab size of the input language
    # output_vocab_size =  vocab size of the output language
    # hidden_size = size of the hidden layer neuron
    # learning rate
    def __init__(self, input_size, output_size, input_vocab_size, output_vocab_size, hidden_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        # initialize hidden and cell states
        self.hidden_state = np.random.random((hidden_size, input_size))
        self.cell_state = np.random.random((hidden_size, input_size))
        # output weight matrix for choosing a word
        self.Weight_output = np.random.random((input_size, output_vocab_size))
        # initialize LSTM cell
        self.LSTM = LSTM(input_size, input_vocab_size, hidden_size, learning_rate)

    # Forward pass for the RNN/LSTM
    def forwardpass(self, input, expected_output):
        self.hidden_state, self.cell_state = self.LSTM.forwardpass(input, self.hidden_state, self.cell_state)

        # compute the Unnormalized probs here with the hidden state.
        y = np.dot(self.hidden_state, self.Weight_output)

        # Apply the softmax to get the normalized probabilities
        probabilities = cf.softmax(y)

        # get the position of the highest predicted prob.
        predicted_index = probabilities.argmax() 
        predicted_output = probabilities[:,predicted_index]

    def backpropagation(self):



