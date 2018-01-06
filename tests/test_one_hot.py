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
input_size = 1
input_vocab_size = 700
hidden_size = 128
output_size = 1
output_vocab_size = 700
input_hidden_dims = input_vocab_size + hidden_size

# Init input and hidden state
input_data = np.zeros((input_size, input_vocab_size))
prev_hidden_state = np.random.random((input_size, hidden_size))
prev_cell_state = np.random.random((input_size, hidden_size))

print(prev_hidden_state.shape[0])
print(prev_hidden_state.shape[1])

# Init Weights for LSTM
Weight_forget = np.random.random((input_hidden_dims, hidden_size))
Weight_igate = np.random.random((input_hidden_dims, hidden_size))
Weight_ogate = np.random.random((input_hidden_dims, hidden_size))
Weight_cell = np.random.random((input_hidden_dims, hidden_size))

# Output weight for choosing a word
Weight_output = np.random.random((hidden_size, output_vocab_size))

# sudo one hot encoded inputs
for i in range(input_size):
    j = np.random.random_integers(input_vocab_size - 1)
    input_data[i,j] = 1

# concat input and hidden state
concat_x_h = np.hstack((input_data, prev_hidden_state))

print("Concatenated input data and hidden state: \n", concat_x_h)

forget_gate = af.sigmoid(np.dot(concat_x_h, Weight_forget))

# input gate calculation sigmoid(W_i * [h_t-1, x_t])
input_gate = af.sigmoid(np.dot(concat_x_h, Weight_igate))

# C prime calculation tanh(W_c * [h_t-1, x_t])
C_prime = af.tanh(np.dot(concat_x_h, Weight_cell))

# output gate calculation sigmoid(W_o[h_t-1, x_t])
output_gate = af.sigmoid(np.dot(concat_x_h, Weight_ogate))

# Cell state calculation (forget gate * C_t-1) + (input gate layer * C̃_t)
cell_state = np.multiply(forget_gate, prev_cell_state) + np.multiply(input_gate, C_prime)

# Hidden state calculation output * tanh(C_t)
hidden_state = np.multiply(output_gate, af.tanh(cell_state))

y = np.dot(hidden_state, Weight_output)

print("Shape of Unnormalized Probabilities: Row: %d, Col: %d" %(y.shape[0], y.shape[1]))

print("Length of y: %d" % len(y))

# instead of doing the softmax over the whole numpy array.
# need to do the softmax over one entry in the numpy array.
probabilities = cf.softmax(y)
print("Normalized probabilities using softmax: \n", probabilities)
print("Sum of probabilities: %d" % np.sum(probabilities))

pos = probabilities.argmax() 
x = probabilities[:,pos] 

print("Highest probabilitiy: %.64f, Position of highest probabilitiy: %d" % (x, pos))


