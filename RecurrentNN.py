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
        self.hidden_state = np.zeros((input_size, hidden_size))
        self.cell_state = np.zeros((input_size, hidden_size))
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
            # print("Hidden state at time step: %d" % t)
            # print(self.hidden_state)

            # reshape the data so that it is a 1 row 700 column 2d array instead of a 1d array
            reshaped_data = input_data[t].reshape(self.input_size, -1)

            self.hidden_state, self.cell_state = self.LSTM.forwardpass(reshaped_data, self.hidden_state, self.cell_state)

            self.hidden_states[t], self.cell_states[t] = self.hidden_state, self.cell_state

            # reshape the hidden state so that it is in the proper dimensional form
            # reshaped_hidden_state = self.hidden_states[t].reshape((self.input_size, self.hidden_size))

            # compute the Unnormalized probs here with the hidden state.
            y = np.dot(self.hidden_state, self.Weight_output)

            print(y)

            # Apply the softmax to get the normalized probabilities
            probabilities = cf.softmax(y)

            # print(probabilities)

            # get the position of the highest predicted prob.
            predicted_outputs[t] = probabilities.argmax()

            # compute the error value
            error = predicted_outputs[t] - expected_output[t].argmax()

            print("States(expected_output: ", expected_output[t].argmax(), " predicted_output: " , predicted_outputs[t])
            print("Print error: ", error)

        return error, predicted_outputs

    # def backpropagation(self):



