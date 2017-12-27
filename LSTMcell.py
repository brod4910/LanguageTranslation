import numpy as np
import ActivationFns as af

class LSTM:

	# where input size is the size of the longest word in the dictionary.
	def __init__(self, input_size, batch_size, hidden_size, expected_output_size, learning_rate):
		self.input_size = input_size
		self.expected_output_size = expected_output_size
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		# Concatenated input and hidden dimensions
		self.input_hidden_dims = hidden_size + input_size
		# Weight Matricies for LSTM
		self.Weight_forget = np.random.random((hidden_size, input_hidden_dims))
		self.Weight_igate = np.random.random((hidden_size, input_hidden_dims))
		self.Weight_outputg = np.random.random((hidden_size, input_hidden_dims))
		self.Weight_cell = np.random.random((hidden_size, input_hidden_dims))
		# gate matricies.
		self.forget_gate = np.zeros((input_size, hidden_size))
		self.input_gate = np.zeros((input_size, hidden_size))
		self.cell_state = np.zeros((input_size, hidden_size))
		self.output_gate = np.zeros((input_size, hidden_size))
		self.hidden_state = np.zeros((input_size, hidden_size))

	# Forward pass for the LSTM cell.
	def forwardpass(input_values, prev_hidden_state, prev_cell_state):
		# concatenated input: x_t and hidden state: h_t-1
		concat_x_h = np.concatenate(input_values, prev_hidden_state)

		# forget gate calculation sigmoid(W_f * [h_t-1, x_t])
		self.forget_gate = af.sigmoid(np.dot(self.Weight_forget, concat_x_h))

		# input gate calculation sigmoid(W_i * [h_t-1, x_t])
		self.input_gate = af.sigmoid(np.dot(self.Weight_igate, concat_x_h))

		# C prime calculation tanh(W_c * [h_t-1, x_t])
		C_prime = af.tanh(np.dot(self.Weight_cell, concat_x_h))

		# output gate calculation sigmoid(W_o[h_t-1, x_t])
		self.output_gate = af.sigmoid(np.dot(self.Weight_outputg, concat_x_h))

		# Cell state calculation (forget gate * C_t-1) + (input gate layer * C̃_t)
		self.cell_state = np.multiply(self.forget_gate, prev_cell_state) + np.multiply(self.input_gate, C_prime)

		# Hidden state calculation output * tanh(C_t)
		self.hidden_state = np.multiply(self.output_gate, af.tanh(self.cell_state))

		# use hidden_state to get the unnormalized prob.
		return self.hidden_state, self.cell_state

	def backpropagation():


