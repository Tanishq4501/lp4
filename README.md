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

## Study notes (theory and viva)

[Viva_Theory.md](Viva_Theory.md) is the general theory sheet (basics, FNN, CNN, autoencoders, CBOW, transfer learning, formulas, traps). Each PS also has its own note next to its notebook: a cell-by-cell walkthrough with the reason for every step, terminology, the theory, a why / why-not table, the results from the notebook, the mistakes met, and viva questions with answers.

| PS | Notes | Topic |
|---|---|---|
| 1A | [1A.md](1A.md) | Feed-forward network on handwritten MNIST (SGD, 11 epochs) |
| 1B | [1B.md](1B.md) | Feed-forward network on Fashion-MNIST (SGD) |
| 1C | [1C.md](1C.md) | Feed-forward network on CIFAR-10 |
| 1E | [1E.md](1E.md) | Feed-forward network on Sign-Language MNIST |
| 2A | [2A.md](2A.md) | Convolutional network on CIFAR-10 |
| 2B | [2B.md](2B.md) | Convolutional network on handwritten MNIST |
| 2C | [2C.md](2C.md) | Convolutional network on Sign-Language MNIST |
| 3A | [3A.md](3A.md) | Credit-card fraud detection with an autoencoder |
| 3B | [3B.md](3B.md) | Anomaly detection on ECG with a convolutional autoencoder |
| 3C | [3C.md](3C.md) | Smart-grid energy anomalies with an LSTM autoencoder |
| 3D | [3D.md](3D.md) | Fault detection in NASA turbofan sensor data with an LSTM autoencoder |
| 3E | [3E.md](3E.md) | Anomaly detection on ECG with a dense autoencoder |
| 3F | [3F.md](3F.md) | Anomaly detection with a convolutional autoencoder on Sign-MNIST (held-out letter) |
| 4A | [4A.md](4A.md) | CBOW word embeddings + classifier for document (review) classification |
| 4B | [4B.md](4B.md) | CBOW (Word2Vec) on the Twitter dataset |
| 4C | [4C.md](4C.md) | CBOW (Word2Vec) on the given text `document3` |
| 4D | [4D.md](4D.md) | Content-based recommender with CBOW embeddings |
| 5A | [5A.md](5A.md) | Transfer learning (VGG16) on Caltech-101 |
| 5B | [5B.md](5B.md) | Transfer learning (VGG16) on Intel Image Classification |
| 5C | [5C.md](5C.md) | Transfer learning (VGG16) on Fashion-MNIST images |
