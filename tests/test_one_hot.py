import os,sys,inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import ActivationFns as af
import CostFns as cf

# (m X n) • (n X p) -> (m X p)
# static params. In practice these would be based on the data.
# hidden size is a hyper param.
input_size = 41
input_vocab_size = 400
hidden_size = 128
output_size = 41
output_vocab_size = 500
input_hidden_dims = input_vocab_size + hidden_size

# Init input and hidden state
input_data = np.zeros((input_vocab_size, input_size))
prev_hidden_state = np.random.random((hidden_size, input_size))
prev_cell_state = np.random.random((hidden_size, input_size))

# Init Weights for LSTM
Weight_forget = np.random.random((hidden_size, input_hidden_dims))
Weight_igate = np.random.random((hidden_size, input_hidden_dims))
Weight_ogate = np.random.random((hidden_size, input_hidden_dims))
Weight_cell = np.random.random((hidden_size, input_hidden_dims))

# Output weight for choosing a word
Weight_output = np.random.random((input_size, output_vocab_size))

# sudo one hot encoded inputs
for j in range(41):
	i = np.random.random_integers(input_vocab_size - 1)
	input_data[i,j] = 1

# concat input and hidden state
concat_x_h = np.concatenate((input_data, prev_hidden_state))

print("Concatenated input data and hidden state: \n", concat_x_h)

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

y = np.dot(hidden_state, Weight_output)

print("Shape of Unnormalized Probabilities: Row: %d, Col: %d" %(y.shape[0], y.shape[1]))

# instead of doing the softmax over the whole numpy array.
# need to do the softmax over one entry in the numpy array.
for j in range(128):
	probabilities = cf.softmax(y[j,:])
	print("Normalized probabilities using softmax: \n", probabilities)
	print("Sum of probabilities: %d" % np.sum(probabilities))

