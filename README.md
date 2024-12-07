I'm trying to learn more about data analysis methods, including artificial intelligence. This repo contains the code related to that.

I am also keeping a private document containing everything I'm learning that I may publicly release in the future.

#### Implemented Projects
- Spam text classifier. Implemented with RNN and LSTM

## RNN

### `single_layer_rnn.py`
This is a single layer rnn that tries to predict the next number. The dataset is the sin function. This isn't great cause there's no noise in the data, so the RNN can overfit as much as it wants, and it's simply interpreted as being "good." 

---

### `multi_layer_rnn.py`
Made a more complex RNN. Has multiple layers and an embedding layer to embed text to an intermediate n-dimensional vector. Its task is to classify whether a given text is spam or not. It didn't take a high value of `embed_size`, `hidden_size`, or number of layers to achieve the maximum score of 86.64%. This proves that unmodified, vanilla RNNs don't scale well to compute.

Conclusion: The upper capabilities of RNNs are reached easily.

## LSTM

### `lstm.py`
Implemented an LSTM with 2 fc connected layers at thje end. Didn't achieve any results better than the RNN implementation. Tried a few different parameters. Reached the same maximum accuracy of 86.64%.

## GAN
### `gan_mnist.py`
This GAN generates hand written numbers using the MNIST dataset. This project could be extended to generate handwritten text.

#### Method 1
Create a GAN for each possible character. Then use the generator of each GAN network to create the individual letters for whatever you want to write.

#### Method 2
Create a GAN network that's trained on all the characters. Then get a really good character classifier (discriminator) to choose which specific character you want to type.

## Utility Files

### `util/unzip_imagenet_train.bat`
The imagenet training data is really weird. It's a zipped folder of zipped folders. So this script unzips all the "mini" folders, assuming that the parent `.tar` folder has already been unzipped. So this script extracts the child `.tar` folders that contain the actual photos.
