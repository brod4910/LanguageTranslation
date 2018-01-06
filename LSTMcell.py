import numpy as np
import ActivationFns as af

class LSTM:

	# where input size is the size of the longest word in the dictionary.
	def __init__(self, input_size, input_vocab_size, hidden_size, learning_rate):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		# Concatenated input and hidden dimensions
		self.input_hidden_dims = hidden_size + input_vocab_size
		# Weight Matricies for LSTM
		Weight_forget = np.random.random((input_hidden_dims, hidden_size))
		Weight_igate = np.random.random((input_hidden_dims, hidden_size))
		Weight_ogate = np.random.random((input_hidden_dims, hidden_size))
		Weight_cell = np.random.random((input_hidden_dims, hidden_size))
		# gate matricies.
		self.forget_gate = np.zeros((input_size, hidden_size))
		self.input_gate = np.zeros((input_size, hidden_size))
		self.cell_state = np.zeros((input_size, hidden_size))
		self.output_gate = np.zeros((input_size, hidden_size))
		# hidden state and cell state intializations.
		self.hidden_state = np.zeros((input_size, hidden_size))
		self.cell_state = np.zeros((input_size, hidden_size))

	# Forward pass for the LSTM cell.
	def forwardpass(input_data, prev_hidden_state, prev_cell_state):
		# concatenated input: x_t and hidden state: h_t-1
		concat_x_h = np.hstack((input_data, prev_hidden_state))

		# forget gate calculation sigmoid(W_f * [h_t-1, x_t])
		self.forget_gate = af.sigmoid(np.dot(concat_x_h, self.Weight_forget))

		# input gate calculation sigmoid(W_i * [h_t-1, x_t])
		self.input_gate = af.sigmoid(np.dot(concat_x_h, self.Weight_igate))

		# C prime calculation tanh(W_c * [h_t-1, x_t])
		C_prime = af.tanh(np.dot(concat_x_h, self.Weight_cell))

		# output gate calculation sigmoid(W_o[h_t-1, x_t])
		self.output_gate = af.sigmoid(np.dot(concat_x_h, self.Weight_ogate))

		# Cell state calculation (forget gate * C_t-1) + (input gate layer * C̃_t)
		self.cell_state = np.multiply(self.forget_gate, prev_cell_state) + np.multiply(self.input_gate, C_prime)

		# Hidden state calculation output * tanh(C_t)
		self.hidden_state = np.multiply(self.output_gate, af.tanh(self.cell_state))

		# use hidden_state to get the unnormalized prob.
		return self.hidden_state, self.cell_state

	def backpropagation():


