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
hidden_size = 4
input_size = 5
vocab_size = 10
input_hidden_dims = hidden_size + input_size

input_data = np.random.random((vocab_size, input_size))
hidden_state = np.random.random((input_size, hidden_size))
output = np.zeros((hidden_size, vocab_size))

concat_x_h = np.concatenate((input_data, hidden_state))

print("Concatenated input and hidden state: \n ", concat_x_h)

Weight_forget = np.random.random((hidden_size, input_hidden_dims))

print("Forget Weights: \n", Weight_forget)

print("Weight dot Concat: \n", np.dot(Weight_forget, concat_x_h))

forget_gate = af.sigmoid(np.dot(Weight_forget, concat_x_h))

print("Forget Gate: \n ", forget_gate)

print("Shape of Forget Gate: Rows: %d , Cols: %d" % (forget_gate.shape[0], forget_gate.shape[1]))


