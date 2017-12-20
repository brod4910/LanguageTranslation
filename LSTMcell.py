import numpy as np
import ActivationFns as af

class LSTM:

	def __init__(self, input_size, expected_output_size, recurrence_length, learning_rate):
		self.input = np.zeros(input_size + expected_output_size)
		self.input_size = input_size + expected_output_size
		self.output = np.zeros(expected_output_size)
		self.cell = np.zeros(expected_output_size)
		self.recurrence_length = recurrence_length
		self.learning_rate = learning_rate
		self.forget_gate = np.random.random((expected_output_size, input_size+expected_output_size))
		self.input_gate = np.random.random((expected_output_size, input_size+expected_output_size))
		self.cell_state = np.random.random((expected_output_size, input_size+expected_output_size))
		self.output_gate = np.random.random((expected_output_size, input_size+expected_output_size))
		self.forget_gradients = np.zeros_like(self.forget_gate)
		self.input_gradients = np.zeros_like(self.input_gate)
		self.cell_state_gradients = np.zeros_like(self.cell_state)
		self.output_gradients = np.zeros_like(self.output_gate)

	
