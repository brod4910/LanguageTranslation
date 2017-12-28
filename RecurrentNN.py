import numpy as np
import os
import LSTMcell
import ActivationFns as af

"""

"""
class RecurrentNeuralNetwork:
    # args: input_size = length of longest sentence, expected_output_size = length of longest output sentence, 
    # learning rate
    def __init__(self, input_size, expected_output_size, hidden_size, learning_rate):
        self.input_size = input_size
        self.expected_output_size = expected_output_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        # initialize hidden and cell states
        self.hidden_state = np.random.random((hidden_size, input_size))
        self.cell_state = np.random.random((hidden_size, input_size))
        # initialize LSTM cell
        self.LSTM = LSTM(input_size, hidden_size, learning_rate)
        self.loss = 0

    # Forward pass for the RNN/LSTM
    def forwardpass(input, expected_output):
        self.hidden_state, self.cell_state = self.LSTM.forwardpass(input, self.hidden_state, self.cell_state)

        # compute the error here witht the hidden state.