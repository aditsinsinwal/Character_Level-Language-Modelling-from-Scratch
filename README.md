This is a character-level RNN implemented from scratch using only NumPy. It reads a text file and trains a Recurrent Neural Network to generate text character-by-character. 


load_data(path)

Purpose: Load and preprocess raw text data.
What it does:Reads the text from input.txt
Creates a vocabulary of unique characters.
Maps each character to an index and vice versa.
Converts the entire text into a list of integer indices.


CharRNN Class
This is a vanilla Recurrent Neural Network (RNN).

Constructor (__init__)
Initializes:

Weights:
Wxh: input to hidden
Whh: hidden to hidden
Why: hidden to output
Biases: bh, by

All weights are initialized randomly with small values.

loss_and_gradients(inputs, targets, h_prev)
Forward pass:
For each character:
One-hot encode the input.
Compute hidden state h[t] using tanh.
Compute output y[t] and softmax probabilities p[t].
Accumulate the cross-entropy loss.

Backward pass (Backpropagation Through Time - BPTT):
Backpropagates the gradients from last time step to first.
Uses tanh derivative: 1 - h²

Clipping is done to prevent exploding gradients.

Returns:

loss: scalar loss

grads: dictionary of gradients

hs[final]: final hidden state to carry forward

sample(h, seed_idx, n)
Text generation:

Given a seed character and hidden state, generates n characters.

Samples characters based on softmax probability distribution (not argmax).

update_parameters(grads)
Standard SGD step:

Updates weights and biases using gradients and learning rate.



train_rnn(...)
Main training loop:

Iterates num_iters times.

Uses a sliding window to get input and target sequences of length seq_length.

Calls loss_and_gradients() to compute forward and backward pass.

Updates weights.

Every 100 steps, prints loss and a generated text sample using sample().

Implements simple gradient descent, no batching, no optimizers, just raw RNN.



__main__ block
Loads text data.

Sets hyperparameters: hidden size, learning rate, etc.

Trains the model.

Generates and prints 500 characters of sample text from a seed character.

