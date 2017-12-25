import numpy as np
import os
import LSTMcell
import ActivationFns as af

"""

"""
class RecurrentNeuralNetwork(object):
    # args: input_size = length of longest word, expected_output_size = length of longest output word, 
    # expected_output = the expected word, learning rate
    def __init__(self, input_size, expected_output_size, expected_output, batch_size, hidden_size,learning_rate):
        self.input_size = input_size
        self.expected_output_size = expected_output_size
        self.expected_output = expected_output
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.Weight_input_hidden = np.random.random((input_size, hidden_size))
        self.Weight_hidden_hidden = np.random.random((hidden_size, hidden_size))
        
        self.hidden_state = np.random.random((batch_size, hidden_size))
        self.cell_state = np.random.random((batch_size, hidden_size))
        # init LSTM cell
        self.LSTM = LSTMcell(input_size, batch_size, hidden_size, expected_output_size, learning_rate)

