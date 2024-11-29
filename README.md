I'm trying to learn more about data analysis methods, including artificial intelligence. This repo contains the code related to that.

I am also keeping a private document containing everything I'm learning that I may publicly release in the future.


## RNN

### `single_layer_rnn.py`
This is a single layer rnn that tries to predict the next number. The dataset is the sin function. This isn't great cause there's no noise in the data, so the RNN can overfit as much as it wants, and it's simply interpreted as being "good." 

---

### `multi_layer_rnn.py`
Made a more complex RNN. Has multiple layers and an embedding layer to embed text to an intermediate n-dimensional vector. Its task is to classify whether a given text is spam or not. It didn't take a high value of `embed_size`, `hidden_size`, or number of layers to achieve the maximum score of 86.64%. This proves that unmodified, vanilla RNNs don't scale well to compute.

Conclusion: The upper capabilities of RNNs are reached easily.

## LSTM

### `lstm.py`
Implemented an LSTM with 2 fc connected layers at thje end. Didn't achieve any results better than the RNN implementation.
