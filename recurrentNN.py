import numpy as np
import os as 
import LSTM.py
import ActivationFns.py as af

class RecurrentNeuralNetwork(object):
    """docstring for RecurrentNeuralNetwork"""
    # args: input (word), expected output, num of words, array of expected outputs, learning rates
    def __init__(self, input_size, expected_output_size, recurrence_length, expected_output_values, learning_rate):
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
        self.hidden_array = np.zeros((recurrence_length+1,expected_output_size))
        # forget gate 
        self.array_forget_gate = np.zeros((recurrence_length+1,expected_output_size))
        # input gate
        self.array_input_gate = np.zeros((recurrence_length+1,expected_output_size))
        # cell state
        self.array_cell_state = np.zeros((recurrence_length+1,expected_output_size))
        # output gate
        self.array_output_gate = np.zeros((recurrence_length+1,expected_output_size))
        # array of expected output values. Using vstack to vertically stack the rows of the output values
        self.expected_output_values = np.vstack(np.zeros(expected_output_values.shape[0]), expected_output_values.T)
        # declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(input_size, expected_output_size, recurrence_length, learning_rate)

    # Forward pass
    def forwardpass(self):
        for i in range(1, recurrence_length+1):
            self.LSTM.input = np.hstack((self.hidden_array[i-1], self.input))
            cell_state, hidden_state, forget_gate, input_gate, cell, output_gate = self.LSTM.forwardProp()
            self.cell_array[i] = cell_state
            self.hidden_array[i] = hidden_state
            self.array_forget_gate[i] = forget_gate
            self.array_input_gate[i] = input_gate
            self.array_cell_state[i] = cell
            self.array_output_gate[i] = output_gate
            self.output_array[i] = af.sigmoid(np.dot(self.weights, hidden_state))
        return self.output_array

    def backProp(self):
        totalError = 0
        deriv_cell_state = np.zeros(self.expected_output_size)
        deriv_hidden_state = np.zeros(self.expected_output_size)
        weight_matrix = np.zeros((self.expected_output_size, self.expected_output_size))
        # LSTM gradients
        forget_gate = np.zeros((self.expected_output_size, self.input_size+self.expected_output_size))
        input_gate = np.zeros((self.expected_output_size, self.input_size+self.expected_output_size))
        cell = np.zeros((self.expected_output_size, self.input_size+self.expected_output_size))
        output_gate = np.zeros((self.expected_output_size, self.input_size+self.expected_output_size))
        for i in range(self.recurrence_length, -1, -1):
            error = self.output_array[i] - self.expected_output[i]
            weight_matrix += np.dot(np.atleast_2d(error * af.derivative_sigmoid(self.output_array[i])), np.atleast_2d(self.hidden_array[i]).T)
            error = np.dot(error, self.weights)
            self.LSTM.cell_state = self.cell_array[i]
            forget_gate_update, input_update, cell_update, output_update, deriv_cell_state, deriv_hidden_state = self.LSTM.backProp(error, self.cell_array[i], self.array_forget_gate[i], self.array_input_gate[i], self.array_cell_state[i], self.array_output_gate[i], deriv_cell_state, deriv_hidden_state)
            totalError += np.sum(error)
            forget_gate += forget_gate_update
            input_gate += input_update
            cell += cell_update
            output_gate += output_update
        self.LSTM.update(forget_gate/self.recurrence_length, input_gate/self.recurrence_length, cell/self.recurrence_length, output_gate/self.recurrence_length)
        self.update(weight_matrix/self.recurrence_length)
        return totalError

    def update(self, update):
        self.RMS = 0.9 * self.RMS + 0.1 * update**2
        self.weights -= self.recurrence_length/np.sqrt(self.RMS + 1e-8) * update
        return


