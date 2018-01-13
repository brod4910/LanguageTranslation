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
        self.output_vocab_size = output_vocab_size
        self.input_vocab_size = input_vocab_size
        # init arrays to store hidden states and cell states
        # self.hidden_states = np.random.uniform(low= -1, high= 1, size=(sequence_length + 1, input_size, hidden_size))
        # self.cell_states = np.random.uniform(low= -1, high= 1, size=(sequence_length + 1, input_size, hidden_size))
        self.hidden_states = np.zeros((sequence_length + 1, input_size, hidden_size))
        self.cell_states = np.zeros((sequence_length + 1, input_size, hidden_size))
        # output weight matrix for choosing a word
        self.Weight_output = np.random.uniform(low= -1, high= 1, size=(hidden_size, output_vocab_size))
        # initialize LSTM cell
        self.LSTM = LSTMcell.LSTM(input_size, input_vocab_size, hidden_size, learning_rate)

    # Forward pass for the RNN/LSTM
    def forwardpass(self, input_data, expected_output):
        # init array for predicted outputs
        predicted_outputs = np.zeros((self.sequence_length, self.input_size, self.output_vocab_size))
        total_loss = 0

        for t in range(1, self.sequence_length + 1):
            # LSTM forwardpass
            self.hidden_states[t], self.cell_states[t] = self.LSTM.forwardpass(input_data[t-1], self.hidden_states[t-1], self.cell_states[t-1])

            # compute the Unnormalized probs here with the hidden state.
            y = np.dot(self.hidden_states[t], self.Weight_output)

            total_loss += cf.cross_entropy_loss(y, expected_output[t-1])

            # Apply the softmax to get the normalized probabilities
            probabilities = cf.softmax(y)

            # get the position of the highest predicted prob.
            predicted_output = probabilities.argmax()

            predicted_outputs[t-1, 0, predicted_output] = 1

            print(total_loss)

        return predicted_outputs, total_loss

    # def backpropagation(self, predicted_outputs, expected_outputs):



