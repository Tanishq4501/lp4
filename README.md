# LP4 practice notebooks

Practice notebooks for the LP-IV (Deep Learning) practical, one notebook per problem statement.

| Notebooks | Topic | Data |
|---|---|---|
| 1A, 1B, 1C, 1E | Feed-forward network (SGD / Adam) | MNIST, Fashion-MNIST, CIFAR-10, Sign-MNIST |
| 2A, 2B, 2C | Convolutional network | CIFAR-10, MNIST, Sign-MNIST |
| 3A-3F | Autoencoder anomaly detection | 3A credit card, 3B conv-AE on ECG, 3C LSTM-AE energy, 3D LSTM-AE NASA turbofan, 3E dense AE on ECG, 3F Sign-MNIST (held-out letter) |
| 4A-4D | CBOW / Word2Vec | 4A IMDB classification, 4B Twitter, 4C document3, 4D recommender |
| 5A-5C | Transfer learning (VGG16) | 5A Caltech-101, 5B Intel images, 5C Fashion-MNIST images |

## Data is not included
The datasets and the VGG16 weights are too large for GitHub, so they are ignored. To run a notebook, put these next to it:
`mnist_train.csv`, `mnist_test.csv`, `fashion-mnist_train.csv`, `fashion-mnist_test.csv`, `cifar-10.csv`, `sign_mnist_train.csv`, `sign_mnist_test.csv` (also in `sg/` for 1E), `creditcard.csv`, `ecg.csv`, `energy_consumption.csv`, `nasa.csv`, `IMDB Dataset.csv`, `twitter_dataset.csv`, `document3.txt`, the folders `Intel Image Classification/`, `caltech-101/`, `fashion mnist/`, and `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`.

Each notebook ends with a small "one example" cell, and 5A-5C have an optional `tr = tr.take(60)` line to speed up training on a CPU.
