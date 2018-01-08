import numpy as np
import os
import LSTMcell
import ActivationFns as af
import CostFns as cf

"""

"""
class RecurrentNeuralNetwork:
    # args: input_size = 1 x input_vocab_size vector, 
    # output_size = 1 x output_vocab_size vector,
    # sequence_length = number of words in the longest sentence in that data for both En and Es 
    # input_vocab_size = vocab size of the input language
    # output_vocab_size =  vocab size of the output language
    # hidden_size = size of the hidden layer neuron
    # learning rate
    def __init__(self, input_size, output_size, sequence_length, input_vocab_size, output_vocab_size, hidden_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        # init array for predicted outputs
        self.predicted_outputs = np.zeros((sequence_length, output_vocab_size))
        # initialize hidden and cell states
        self.prev_hidden_state = np.random.random((input_size, hidden_size))
        self.prev_cell_state = np.random.random((input_size, hidden_size))
        # init arrays to store hidden states and cell states
        self.hidden_states = np.zeros((sequence_length, hidden_size))
        self.cell_states = np.zeros((sequence_length, hidden_size))
        # output weight matrix for choosing a word
        self.Weight_output = np.random.random((hidden_size, output_vocab_size))
        # initialize LSTM cell
        self.LSTM = LSTMcell.LSTM(input_size, input_vocab_size, hidden_size, learning_rate)

    # Forward pass for the RNN/LSTM
    def forwardpass(self, input_data, expected_output):
        predicted_outputs = np.zeros((self.sequence_length, 1))

        for t in range(self.sequence_length):
            # This line is a bit hacky. I am assigning the previous states to this function, because if I assigned
            # the states returned from this function to the array I would have to reshape the prev states every time            
            self.prev_hidden_state, self.prev_cell_state = self.LSTM.forwardpass(input_data[t,:], self.prev_hidden_state, self.prev_cell_state)

            self.hidden_states[t], self.cell_states[t] = self.prev_hidden_state, self.prev_cell_state

            # reshape the hidden state so that it is in the proper dimensional form
            reshaped_hidden_state = self.hidden_states[t].reshape((self.input_size, self.hidden_size))

            # compute the Unnormalized probs here with the hidden state.
            y = np.dot(reshaped_hidden_state, self.Weight_output)

            # Apply the softmax to get the normalized probabilities
            probabilities = cf.softmax(y)

            # get the position of the highest predicted prob.
            predicted_outputs[t] = probabilities.argmax()

            # compute the error value
            error = predicted_outputs[t] - expected_output[t]

            print("States(expected_output: ", expected_output.argmax(), " predicted_output: " , predicted_outputs[t])

        return error, predicted_outputs, hidden_states, cell_states

    # def backpropagation(self):



