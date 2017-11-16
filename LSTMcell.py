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

	def forwardProp(self):
		forget_gate = af.sigmoid(np.dot(self.forget_gate, self.input))
		self.cell_state *= forget_gate
		input_gate = self.sigmoid(np.dot(self.input_gate, self.input))
		cell = af.tanh(np.dot(self.cell, self.input))
		self.cell_state += input_gate * cell
		output_gate = self.sigmoid(np.dot(self.output_gate, self.input))
		self.output = output_gate * af.tanh(self.cell_state)
		return self.cell_state, self.output, forget_gate, input_gate, cell, output_gate

	def backwardProp(self, error, previous_cell_state, forget_gate, input_gate, cell, output_gate, 
		deriv_forget_cell_state, deriv_forget_hidden_state):
		error = np.clip(error + deriv_forget_hidden_state, -6, 6)
		deriv_output = af.tanh(self.cell_state) * error
		output_update = np.dot(np.atleast_2d(deriv_output * af.derivative_tanh(output_gate)).T, np.atleast_2d(self.input))
		deriv_cell_state = np.clip(error * output_gate * af.derivative_tanh(self.cell_state) + deriv_forget_cell_state, -6, 6)
		deriv_cell = deriv_cell_state * input_gate
		cell_update = np.dot(np.atleast_2d(deriv_cell * af.derivative_tanh(cell)).T, np.atleast_2d(self.input))
		deriv_input = deriv_cell_state * cell
		input_update = np.dot(np.atleast_2d(deriv_input * af.derivative_sigmoid(input_gate)).T, np.atleast_2d(self.input))
		deriv_forget_gate = deriv_cell_state * previous_cell_state
		forget_gate_update = np.dot(np.atleast_2d(deriv_forget_gate * af.derivative_sigmoid(forget_gate)).T, np.atleast_2d(self.input))
		deriv_previous_cell_state = deriv_cell_state * forget_gate
		deriv_hidden_state = np.dot(deriv_cell, self.cell)[:self.expected_output_size] + np.dot(deriv_output, self.output_gate)[:self.expected_output_size] + np.dot(deriv_input, self.input_gate)[:self.expected_output_size] + np.dot(deriv_forget_gate, self.forget_gate)[:self.expected_output_size]
		return forget_gate_update, input_update, cell_update, output_update, deriv_previous_cell_state, deriv_hidden_state

	def update(self, forget_gate_update, input_update, cell_update, output_update):
		self.forget_gradients = 0.9 * self.forget_gradients + 0.1 * forget_gate_update**2
		self.input_gradients = 0.9 * self.input_gradients + 0.1 * input_gradients**2
		self.cell_state_gradients = 0.9 * self.cell_state_gradients + 0.1 * cell_state_gradients**2
		self.output_gradients = 0.9 * self.output_gradients + 0.1 * output_gradients**2

		self.forget_gate -= self.learning_rate/np.sqrt(self.forget_gradients + 1e-8) * forget_gate_update
		self.input_gate -= self.learning_rate/np.sqrt(self.input_gradients + 1e-8) * input_update
		self.cell_state -= self.learning_rate/np.sqrt(self.cell_state_gradients + 1e-8) * cell_update
		self.output_gate -= self.learning_rate/np.sqrt(self.output_gradients + 1e-8) * output_update
		return


