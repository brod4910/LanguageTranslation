import numpy as np
import os as 
import LSTM.py

class RecurrentNeuralNetwork(object):
    """docstring for RecurrentNeuralNetwork"""
    # args: input (word), expected output, num of words, array of expected outputs, learning rates
    def __init__(self, input_size, expected_output_size, recurrence_length, eo, learning_rate):
        # intiailize input (first word)
        self.input = np.zeros(input_size)
        # size of input
        self.input_size = input_size
        # expected output 
        self.expected_output = np.zeros(expected_output_size)
        # size of output
        self.expected_output_size = expected_output_size
        # initialize weight matrix for LSTM cell
        self.weights = np.random.random((expected_output_size, expected_output_size))
        # matrix used for root mean square propagation
        self.RMS = np.zeros_like(self.weights)
        # length of reccurent neural network - number of recurrences i.e number of words
        self.recurrence_length = recurrence_length
        # learning rate
        self.learning_rate = learning_rate
        # array for storing inputs
        self.input_array = np.zeros((recurrence_length+1, input_size))
        # array for storing cell states
        self.cell_array = np.zeros((recurrence_length+1,expected_output_size))
        # array for storing outputs
        self.output_array = np.zeros((recurrence_length+1,expected_output_size))
        # array for storing hidden states
        self.hiddn_array = np.zeros((recurrence_length+1,expected_output_size))
        # forget gate 
        self.array_forget_gate = np.zeros((recurrence_length+1,expected_output_size))
        # input gate
        self.array_input_gate = np.zeros((recurrence_length+1,expected_output_size))
        # cell state
        self.array_cell_state = np.zeros((recurrence_length+1,expected_output_size))
        # output gate
        self.array_output_gate = np.zeros((recurrence_length+1,expected_output_size))
        # array of expected output values
        self.expected_output_values = np.vstack((np.zeros(expected_output_values.shape[0]), expected_output_values.T))
        # declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(input_size, expected_output_size, recurrence_length, learning_rate)










