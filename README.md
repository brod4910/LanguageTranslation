# LanguageTranslation_IPFS

Language Translation application using IPFS

Simple application that uses a recurrent neural network to translate from English to Spanish and vice versa.
Uses the Ethereum block chain to deploy the application to the IPFS.
Application will also be deployable to run on your localhost.

# Floyd

Using Floyd to train RNN on the cloud.

# Basic Structure (Sequence-to-sequence based model)

The idea for this application is to use two recurrent neural networks

(English input sentence) - Encoding RNN/LSTM - (encoded sentence) - Decoding RNN/LSTM - (output in spanish)

## Rough idea

The idea is that we train the first neural network to encode the input sentence, but take the sentence word by word until we have a fully encoded sentence. Next take the encoded sentence to feed into the next RNN that is trained to translate from the encoded sentence to decode the sentence into spanish and vice versa.

# Frameworks

Will not be using frameworks such as tensorflow, pytorch, etc.
Mainly for raw neural network practice.
Will be using pickle to save weights of both networks after training.
Will not be using numpy...(just kidding).
Using Ethereum block chain (IPFS). (Not a framework but might as well add that in)
Decided to use the Google Translate API for training purposes. Could not find reliable sources of dictionaries and translations.

# Asides
(12/20)
Cleaned the data I retrived from using Google's translate api.

(12/21)
Some noteable things during my research, have been that the RNN portion of our network is only concerned about the inputs to a cell, the hidden state output and the cell state output.

Whereas an LSTM cell is concerned about 
input: input, previous cell state input, previous hidden state input.
intermediate: forget gate = sigmoid(W_f * [h_t-1, x_t] + b_f), 
			  input gate layer = sigmoid(W_i * [h_t-1, x_t] + b_i),
			  C̃_t = tanh(W_c * [h_t-1, x_t] + b_c),
			  C_t = (forget gate * C_t-1) + (input gate layer * C̃_t),
			  output = sigmoid(W_o[h_t-1, x+t] + b_o),
			  hidden state = output * tanh(C_t)


