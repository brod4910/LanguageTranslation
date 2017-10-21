# LanguageTranslation_IPFS
Language Translation application using IPFS

Simple application that uses a recurrent neural network to translate from English to Spanish.
Uses the Ethereum block chain to deploy the application to the IPFS.

# Floyd

Using Floyd to train RNN on the cloud.

# Basic Structure (Sequence-to-sequence based model)

The idea for this application is to use two recurrent neural networks

(English input sentence) - Encoding RNN/LSTM - (encoded sentence) - Decoding RNN/LSTM - (output in spanish)

## Rough idea

The idea is that we train the first neural network to encode the input sentence, but take the sentence word by word until we have a fully encoded sentence. Next take the encoded sentence to feed into the next RNN that is trained to translate from the encoded sentence to decode the sentence into spanish.

# Frameworks

Will not be using frameworks such as tensorflow, pytorch, etc.
Mainly for raw neural network practice.
Will be using pickle to save weights of both networks after training.
Will not be using numpy...(just kidding).
Using Ethereum block chain (IPFS). (Not a framework but might as well add that in)