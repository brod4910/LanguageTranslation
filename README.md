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
(12/20/2017)
Cleaned the data I retrived from using Google's translate api.

(12/21/2017)
Some noteable things during my research, have been that the RNN portion of our network is only concerned about the inputs to a cell, the hidden state output and the cell state output.

Whereas an LSTM cell is concerned about <br />
input: input, previous cell state input, previous hidden state input. <br />
intermediate: <br />
			  forget gate = **sigmoid(W_f * [h_t-1, x_t] + b_f)**, <br />
			  input gate layer = **sigmoid(W_i * [h_t-1, x_t] + b_i)**, <br />
			  C̃_t = **tanh(W_c * [h_t-1, x_t] + b_c)**, <br />
			  C_t = **(forget gate * C_t-1) + (input gate layer * C̃_t)**, <br />
			  output = **sigmoid(W_o[h_t-1, x_t] + b_o)**, <br />
			  hidden state = **output * tanh(C_t)**

Input to the neural net will be the length of the longest word in the dictionary. All other words not of this length
will be padded with zeros.

Dimensions of the LSTM matricies may not be correct.

(12/24/2017)
Still unsure about the dimensions of the LSTM matricies.
Currently am figuring out if the LSTM cell should handle its own calculations for the loss function or if the RNN class should handle that. I think the LSTM cell should only handle what it requires. The RNN should do the heavy lifting since the LSTM is just a neuron in theory.

Just realized that I was thinking in word length and not sentence length. This now changes the whole perspective I thought I was going for. Minor refactoring needs to be done.

(12/26/2017)
After further analysis, I will be using pre-trained word2vec models for my training data and target data. Since the idea behind this model will be used for sentence translation.

An optimization that has come to my attention is that we can concatenate h_t-1 and x_t, thus negating the previous implementation I had. This way instead of instantiating Weight vectors for gates that connect to h_t-1 and x_t it will be <br />k x (h_t-1 + x_t), where k are the gate dimensions and (h_t-1 + x_t) are input and hidden state dimensions respectively.

Another realization is that some of the gates more notably the hidden and cell state gates use the hadamard product. Instead of using the dot product we want to do an entry-wise product when computing the result


(12/27/2017) 
After further analysis I have come to the conclusion that the LSTM cell should know nothing about the input until it does the forward pass. The only time it will know anything about the input is when we intialize the LSTM cell. This way there is no confusion about the dimensions.

After further testing I have realized what each dimension of each matrix in the LSTM cell should be. Example code is in the test.py located in tests.


(1/3/2018)
Need to derive the gradients by hand or I will not understand the implementation in code.

I am suspicious of my code, since I am currently not storing the hidden states of the previous time steps.
Meaning that I cannot compute the gradients of the hidden states at the previous time steps.

I also need to store the outputs of the model, so that I can compute the respective gradients.

(1/4/2018)
Having trouble understanding the final output to my neural net implementation.
The probabilites do not make sense to me. I am running some tests in test.py.

Realized that my tests have been little to no help, since I am using a random input from 0 < x < 1.

Currently creating sudo one hot encoded inputs to further test my doubts and suspicions.

Could have structured my tests differently, but I find it useful to engrain and repeat the code over. This way I can grasp the concepts and explain them later on.

(1//5/2018)
After further testing, I have figured out that I had the dimensions wrong. I was suspicious when i took the softmax of the output I had originally. The dimensions of the output y: 128x700, which did not make sense since in theory if the input size was 41 then the output should have been 41x700. It turns out that I had the wrong picture of the matricies in my head. I put the hidden size in as the rows and the input size as the columns. This lead to the wonky dimensions.

(1/6/2018)
Since I need to backpropagate through time, I need to save the previous hidden states, cell states, expected outputs, predicted outputs and inputs. I am unsure about how to add a sequence length and how it will change the current state of my code.