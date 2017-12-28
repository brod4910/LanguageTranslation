import os,sys,inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import ActivationFns as af

# # sanity check for activation functions
# Weight_input_forget = np.random.random((10, 5))

# forget_gate = np.zeros((10, 5))

# print(Weight_input_forget)

# forget_gate = af.sigmoid(Weight_input_forget)

# print(forget_gate)

# sanity check for dimensions of weights and gates
hidden_size = 128
input_size = 41
vocab_size = 10
input_hidden_dims = hidden_size + vocab_size

input_data = np.random.random((vocab_size, input_size))
hidden_state = np.random.random((hidden_size, input_size))
prev_cell_state = np.random.random((hidden_size, input_size))

concat_x_h = np.concatenate((input_data, hidden_state))

print("Concatenated input and hidden state: \n ", concat_x_h)

print("Shape of concat_x_h: Row: %d, Col: %d" %(concat_x_h.shape[0], concat_x_h.shape[1]))

Weight_forget = np.random.random((hidden_size, input_hidden_dims))
Weight_igate = np.random.random((hidden_size, input_hidden_dims))
Weight_ogate = np.random.random((hidden_size, input_hidden_dims))
Weight_cell = np.random.random((hidden_size, input_hidden_dims))

forget_gate = af.sigmoid(np.dot(Weight_forget, concat_x_h))

# input gate calculation sigmoid(W_i * [h_t-1, x_t])
input_gate = af.sigmoid(np.dot(Weight_igate, concat_x_h))

# C prime calculation tanh(W_c * [h_t-1, x_t])
C_prime = af.tanh(np.dot(Weight_cell, concat_x_h))

# output gate calculation sigmoid(W_o[h_t-1, x_t])
output_gate = af.sigmoid(np.dot(Weight_ogate, concat_x_h))

# Cell state calculation (forget gate * C_t-1) + (input gate layer * C̃_t)
cell_state = np.multiply(forget_gate, prev_cell_state) + np.multiply(input_gate, C_prime)

# Hidden state calculation output * tanh(C_t)
hidden_state = np.multiply(output_gate, af.tanh(cell_state))

print("Shape of forget_gate: Row: %d, Col: %d" %(forget_gate.shape[0], forget_gate.shape[1]))
print("Shape of input_gate: Row: %d, Col: %d" %(input_gate.shape[0], input_gate.shape[1]))
print("Shape of C_prime: Row: %d, Col: %d" %(C_prime.shape[0], C_prime.shape[1]))
print("Shape of output_gate: Row: %d, Col: %d" %(output_gate.shape[0], output_gate.shape[1]))

print("Shape of hidden_state: Row: %d, Col: %d" %(hidden_state.shape[0], hidden_state.shape[1]))
print("Shape of cell_state: Row: %d, Col: %d" %(cell_state.shape[0], cell_state.shape[1]))





