import os,sys,inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import LSTMcell
import RecurrentNN

# def __init__(self, input_size, output_size, sequence_length, input_vocab_size, output_vocab_size, hidden_size, learning_rate):

def main():
	# (m X n) • (n X p) -> (m X p)
	# static params. In practice these would be based on the data.
	# hidden size is a hyper param.
	input_size = 1
	input_vocab_size = 50
	hidden_size = 128
	output_size = 1
	output_vocab_size = 50
	input_hidden_dims = input_vocab_size + hidden_size
	sequence_length = 5
	learning_rate = .0001

	# predicted outputs
	predicted_outputs = np.zeros((sequence_length, input_size,output_vocab_size))

	# Init sudo input data and sudo output data
	input_data = np.zeros((sequence_length, input_size,input_vocab_size))
	output_data = np.zeros_like((input_data))

	# sudo one hot encoded inputs
	for i in range(sequence_length):
	    k = np.random.random_integers(input_vocab_size - 1)
	    input_data[i,0,k] = 1
	    k = np.random.random_integers(output_vocab_size - 1)
	    output_data[i,0,k] = 1

	# print("input data row: %d, col: %d" % (input_data.shape[0], input_data.shape[1]))

	RNN = RecurrentNN.RecurrentNeuralNetwork(input_size, output_size, sequence_length, input_vocab_size, output_vocab_size, hidden_size, learning_rate)

	predicted_output = RNN.forwardpass(input_data, output_data)

	print(predicted_output)

if __name__ == "__main__":
	main()


