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
		# Weight Matricies for LSTM
		self.Weight_input_forget = np.random.random((input_size, hidden_size))
		self.Weight_hidden_forget = np.random.random((hidden_size, hidden_size))
		self.Weight_hidden_igate = np.random.random((hidden_size, hidden_size))
		self.Weight_input_igate = np.random.random((input_size, hidden_size))
		self.Weight_hidden_cell = np.random.random((hidden_size, hidden_size))
		self.Weight_input_cell = np.random.random((input_size, hidden_size))
		self.Weight_hidden_output = np.random.random((hidden_size, hidden_size))
		self.Weight_input_output = np.random.random((input_size, hidden_size))
		# gate matricies
		self.forget_gate = np.zeros((batch_size, hidden_size))
		self.input_gate = np.zeros((batch_size, hidden_size))
		self.cell_state = np.zeros((batch_size, hidden_size))
		self.output_gate = np.zeros((batch_size, hidden_size))
		self.hidden_state = np.zeros((batch_size, hidden_size))

	def forwardpass(input_values, prev_hidden_state, prev_cell_state):
