# LP4 viva: theory you need (organised like your PS)

How to read each topic: first the **one-line answer** (say this first), then **if they dig deeper** (the extra detail). Numbers in *italics* are from your own notebooks, so you can quote them.

## 0. Which topics belong to which PS

| Topic | PS | Section |
|---|---|---|
| Basics (loss, optimiser, metrics, over-fitting) | all | 1 |
| Feed-forward network (FNN) | 1A 1B 1C 1E | 2 |
| Convolutional network (CNN) | 2A 2B 2C | 3 |
| Autoencoders, anomaly detection (dense, conv, LSTM) | 3A-3F | 4 |
| Word embeddings, CBOW, recommender | 4A-4D | 5 |
| Transfer learning (VGG16) | 5A 5B 5C | 6 |
| Datasets, your results, formulas, traps | all | 7-10 |

---
## 1. Basics every viva starts with

### 1.1 Neural network, training, backpropagation
**One line:** a network is layers of weighted sums plus non-linear activations; training adjusts the weights to make the loss small.
**Deeper:**
- Neuron: `output = activation(w1*x1 + w2*x2 + ... + b)`. A layer is many neurons side by side.
- **Forward pass:** input -> layers -> prediction. **Loss:** one number saying how wrong the prediction is.
- **Backpropagation:** uses the chain rule to compute how the loss changes with every weight (the gradient). **Gradient descent** then moves each weight a small step against its gradient: `w = w - lr * gradient`.
- Without non-linear activations, many layers collapse into a single linear map, so the network could not learn curves.

### 1.2 Activation functions (what you used and why)

| Activation | Formula idea | Where you used it | Why |
|---|---|---|---|
| **ReLU** | `max(0, x)` | hidden layers everywhere | cheap, does not saturate for positive values, fights vanishing gradients |
| **Softmax** | turns scores into probabilities that sum to 1 | last layer of 1A-2C, 5A-5C | one probability per class |
| **Sigmoid** | squashes to 0..1 | 4A last layer (binary); 3E/3B/3F autoencoder output | a single probability; or data scaled to 0..1 |
| **Linear** (none) | output = input | 3A autoencoder output, 3C/3D `Dense` | standardised data can be negative or large |

### 1.3 Loss functions

| Loss | Task | Your PS |
|---|---|---|
| **Sparse categorical cross-entropy** `-log(p of the true class)` | multi-class with integer labels | 1A-2C, 5A-5C |
| **Binary cross-entropy** | two classes, sigmoid output | 4A |
| **MSE** (mean squared error) | reconstruct numbers | 3A-3F |

`sparse_` vs plain `categorical_crossentropy`: same maths; sparse takes integer labels (`7`), plain takes one-hot labels (`[0,0,0,0,0,0,0,1,0,0]`).

### 1.4 Optimisers
- **SGD:** `w -= lr * gradient`. One global learning rate. Simple, slower, sensitive to the rate. Keras default `lr = 0.01` (1A and 1B use it; *94.29 % and 85.55 %*).
- **Adam:** keeps running averages of the gradient (momentum) and of its square (scaling), so each weight gets its own adaptive step. Default `lr = 1e-3`. Usually faster and needs less tuning.
- **Learning rate:** too large -> loss jumps or diverges; too small -> very slow. That is why fine-tuning uses `1e-5`.

### 1.5 Epoch, batch, iteration (with your numbers)
- **Epoch** = one pass over the whole training set. **Batch** = samples used for one weight update. **Iteration** = one update.
- 1A: 60,000 images, batch 128 -> `60000/128` = **469 iterations per epoch** (your log shows `469/469`).

### 1.6 Over-fitting
**One line:** the model memorises the training data and does worse on new data.
**Signs:** training accuracy keeps rising, validation accuracy stalls, validation loss starts to rise. *Your 1E: train 99 %, validation 72 %; your 2C: train 100 %, validation 89 %.*
**Fixes:** more data or augmentation, Dropout, fewer epochs, `EarlyStopping`, a smaller model, weight decay.
**Under-fitting:** poor on both train and validation (model too small or trained too little).
**Dropout:** randomly switches off a fraction of units during training only, so the network cannot rely on single units.
**Data leakage:** information from the test set leaks into training (for example fitting a scaler on all data, building windows across engines, training Word2Vec before splitting).

### 1.7 Metrics
| Metric | Meaning |
|---|---|
| **Accuracy** | correct / total. Misleading with imbalance |
| **Precision** | of what I flagged, how much was truly positive: `TP / (TP + FP)` |
| **Recall** | of the true positives, how many I found: `TP / (TP + FN)` |
| **F1** | harmonic mean of precision and recall |
| **Confusion matrix** | rows = true, columns = predicted |
| **ROC-AUC** | probability that a random anomaly gets a higher score than a random normal sample. 0.5 = guessing, 1.0 = perfect. Needs no threshold |

Imbalance example (3A): 99.83 % of transactions are normal, so "always normal" scores 99.83 % accuracy and finds no fraud. Use precision, recall, F1 and AUC.

### 1.8 Scaling
- **/255** for pixels (0-255 -> 0-1): small, similar-sized inputs give stable, faster training.
- **MinMaxScaler** (0..1) when the output layer is sigmoid. **StandardScaler** (mean 0, std 1) when the output is linear.
- **Fit the scaler on training data only**, then apply it to test data (otherwise leakage). In anomaly detection, fit it on **normal training rows only**.

---
## 2. FNN (1A, 1B, 1C, 1E)

**One line:** a feed-forward network flattens the image into one long vector and passes it through dense layers; the last layer's softmax gives class probabilities.

**Your architecture:** `Flatten -> Dense(256, relu) -> Dense(128, relu) -> Dense(classes, softmax)`.

**Parameter count (be able to do this):** a `Dense` layer has `inputs x outputs + outputs` parameters.
1A: `784*256+256 = 200,960`; `256*128+128 = 32,896`; `128*10+10 = 1,290`; **total 235,146**. Check with `model.summary()`.

**Why the output layer size matters:** it must be `max label + 1`. Digits and Fashion: 10. Sign-MNIST: labels run 0-24 (9 never occurs), so **25**. With 10 outputs you get "label 24 is outside the valid range".

**Why an FNN does badly on images (1C: *46.74 %*; CNN in 2A: *69.69 %*):** flattening throws away where pixels are; a 32x32x3 image becomes 3,072 independent numbers, there are many parameters, and there is no translation invariance (the same object in a different place looks totally different).

**SGD vs Adam on MNIST:** both work; SGD needs more epochs. 1A used SGD for 11 epochs (*94.29 %*); a CNN with Adam reaches *98.9 %* in 5 (2B).

**Typical questions:**
- *Why softmax at the end?* It outputs a probability distribution over classes.
- *Why not one-hot labels here?* We use the sparse loss, which accepts integers.
- *Why is validation = test in your notebooks?* The PS gives only train and test files; stated honestly, the reported number is a test score. With CIFAR (one file) we made an 80/20 split.

---
## 3. CNN (2A, 2B, 2C)

**One line:** a CNN slides small filters over the image to detect local patterns (edges, textures), stacks them into richer patterns, and shares the same filter weights everywhere.

**Building blocks**
- **Convolution:** a filter (for example 3x3) moves over the image computing a dot product; the result is a **feature map**. `Conv2D(32, 3)` learns 32 filters.
- **Padding:** `valid` (none, the map shrinks) or `same` (zero-pad so size is kept at stride 1).
- **Stride:** how many pixels the filter jumps; stride 2 halves the size.
- **Pooling (`MaxPooling2D`):** keeps the maximum of each 2x2 block: halves width and height, cuts computation, gives small shift invariance.
- **Flatten + Dense:** turn the final feature maps into a vector, then classify.

**Size formulas**
- `valid`: `out = (n - k) / stride + 1`. `same`: `out = ceil(n / stride)`.
- **Conv parameters:** `(k * k * channels_in + 1) * filters`.

**Worked example, your 2B (MNIST, no padding):**

| Layer | Output | Parameters |
|---|---|---|
| input | 28x28x1 | 0 |
| `Conv2D(32,3)` | 26x26x32 | `(3*3*1+1)*32 = 320` |
| `MaxPooling2D` | 13x13x32 | 0 |
| `Conv2D(64,3)` | 11x11x64 | `(3*3*32+1)*64 = 18,496` |
| `MaxPooling2D` | 5x5x64 | 0 |
| `Flatten` | 1,600 | 0 |
| `Dense(128)` | 128 | `1600*128+128 = 204,928` |
| `Dense(10)` | 10 | 1,290 |
| **Total** | | **225,034** |

**Why a CNN beats an FNN on images:** local connectivity, weight sharing (far fewer parameters per feature), translation invariance, and a hierarchy of features (edges -> shapes -> objects).

**Your results:** 2B *98.88 %*, 2C *89.26 %* (over-fits, train 100 %), 2A *69.69 %* (CIFAR is harder: colour, clutter, 32x32 pixels).

**Typical questions:** why pooling; what a filter learns; why relu after conv; how to reduce over-fitting in 2C (Dropout, augmentation, early stopping); why CNN has fewer parameters than the FNN for the same image.

---
## 4. Autoencoders and anomaly detection (3A-3F)

### 4.1 What an autoencoder is
**One line:** a network trained to reproduce its input through a narrow middle layer, so it must learn a compressed representation.
- **Encoder** compresses input to the **bottleneck (latent code)**; **decoder** rebuilds the input from the code. Loss = reconstruction error (MSE). Unsupervised: input = target, `fit(X, X)`.
- It must be **undercomplete** (bottleneck smaller than input). If the bottleneck is as big as the input, the network can just copy, and every sample, anomalies included, is rebuilt perfectly. *Your 3B and 3F showed this: without the Flatten -> Dense bottleneck AUC dropped (3B 0.98 -> 0.93, 3F 0.80 -> 0.58).*
- With linear activations an autoencoder behaves like **PCA**; non-linear ones can learn curved structure.

### 4.2 Anomaly detection with it
1. **Train on normal data only**, so it learns what "normal" looks like.
2. For each new sample compute the **reconstruction error** (mean squared difference between input and output).
3. **Normal samples** are rebuilt well (small error); **anomalies** are not (large error).
4. Pick a **threshold**; error above it means anomaly.

**Threshold choices:** `mean + 3*std` of the normal training errors (strict, few false alarms, may miss some; 3 sigma covers 99.7 % only if errors are bell-shaped, which they rarely are), or a **percentile** (95th, 99th), or the value that gives the best F1 on validation data. It is always a precision-recall trade-off: higher threshold -> fewer false alarms (precision up) but more missed anomalies (recall down).

**Evaluation:** never plain accuracy. Report precision, recall, F1 of the anomaly class, the confusion matrix and **ROC-AUC**. The key plot is the error histogram of normal vs anomaly with the threshold line.

### 4.3 Preprocessing rules
- Scaler fitted on **normal training rows only**; the same scaler for the test set.
- Test data goes through the same preprocessing. Evaluating on raw data instead of the scaled array flags every sample (your 3E bug).
- **Error axis = every axis except the first:** `(N,30)` -> `axis=1`; `(N,140,1)` -> `(1,2)`; `(N,28,28,1)` -> `(1,2,3)`.

### 4.4 Each PS

| PS | Data and what is the anomaly | Model | Notes and your result |
|---|---|---|---|
| **3A** | Credit-card transactions (284,807 rows, 492 frauds = 0.17 %). Features V1-V28 are PCA outputs, plus Time, Amount. Fraud = anomaly | Dense 30-24-14-7-14-24-30, **linear** output, `StandardScaler`, threshold mean+3std | Extreme imbalance. *AUC 0.93, recall 0.82, precision 0.22* |
| **3E** (ECG) | ECG heartbeats, 140 points each. **Label 1 = normal, 0 = abnormal** in this file | Dense 140-64-32-16-32-64-140, **sigmoid** output, `MinMaxScaler`, 95th percentile | *AUC about 0.99, F1 about 0.96* |
| **3B** (ECG) | same data | **Conv1D** encoder/decoder with a Flatten -> Dense(16) bottleneck | Beat is a 1-D signal, so a shared 5-sample filter learns waveform shapes. *AUC about 0.98* |
| **3F** | Sign-MNIST; letter Y (label 24) is held out of training and is the anomaly | **Conv2D / Conv2DTranspose**, Dense(32) bottleneck | Hard: Y looks like the others, and test errors are about twice the training errors (the test images differ from the training ones). *AUC 0.83, F1 0.18* |
| **3C** | Smart-grid energy, 5,000 half-hourly rows, 5 features, `Anomaly_Label` | **LSTM autoencoder**, windows of 10, time-ordered 70/30 split | Anomalies are random spikes. *AUC 0.69, recall 0.23* |
| **3D** | NASA turbofan, 100 engines run to failure, 21 sensors; fault = last 30 cycles (RUL <= 30) | **LSTM autoencoder**, windows of 30, split **by engine**, smoothing, 99th percentile | Drop `RUL` (it defines the label: leakage). *AUC 0.92, precision 0.89, recall 0.46* |

### 4.5 Conv and LSTM autoencoder details
- **Conv1D / Conv2D encoder:** strided convolutions shrink the signal or image. **Conv1DTranspose / Conv2DTranspose** (up-sampling convolutions) grow it back. Final layer: sigmoid for 0..1 data.
- **LSTM:** a recurrent layer with a cell state and three gates (input, forget, output) that decide what to remember. It captures **time dependence** and avoids the vanishing gradient of a plain RNN.
- **LSTM autoencoder:** `LSTM(32)` encodes the whole window into one vector; **`RepeatVector(W)`** copies that vector W times (once per time step) as input to the decoder; `LSTM(32, return_sequences=True)` unrolls it; **`TimeDistributed(Dense(features))`** applies the same Dense layer to each time step.
- **Windowing:** an LSTM needs fixed-length sequences, so slide a window of W rows over the series (`a[i:i+W]`). Window label: any anomalous step (3C) or the last step (3D). **Never build windows across two engines** and split by engine to avoid leakage from overlapping windows.
- **Why LSTM for sensors and energy:** consecutive readings depend on each other; a dense autoencoder treats every time step independently.

### 4.6 Typical questions
- Why train only on normal data? Why not accuracy? How did you choose the threshold, and what does raising it do?
- What if the bottleneck equals the input size?
- Why `sigmoid` at the output in 3E but linear in 3A? (data scaled 0-1 versus standardised data that can be negative)
- Why is 3F so much weaker than 3E? Why is 3C weak?
- Autoencoder vs PCA? Other variants (name them): denoising (input corrupted, target clean), sparse, variational (learns a probability distribution).
- Real uses: fraud, machine health monitoring, medical signals, denoising, compression.

---
## 5. Word embeddings, CBOW, recommender (4A-4D)

### 5.1 Why embeddings
**One line:** a word embedding is a dense vector (for example 100 numbers) in which words used in similar contexts get similar vectors.
- **One-hot** vectors are huge (vocabulary size), mostly zeros, and say nothing about similarity. Embeddings are small and capture meaning ("great" is near "excellent").

### 5.2 Word2Vec and CBOW
- **Word2Vec** (Mikolov, 2013) has two architectures:
  - **CBOW (Continuous Bag of Words), `sg=0`:** the **context** words (the window around a position) are looked up in the embedding matrix and **averaged**; a softmax layer predicts the **centre word**. Fast, good for frequent words.
  - **Skip-gram, `sg=1`:** the centre word predicts each context word. Slower, better for rare words.
- After training, the **embedding matrix rows are the word vectors**. The prediction task is only a way to learn them.
- Speed trick: gensim uses **negative sampling** (default 5 negatives) instead of a softmax over the whole vocabulary.

**Parameters:** `vector_size` (dimension, 100/50), `window` (context size, 5), `min_count` (ignore words rarer than this; 2 for tweets, 1 for the tiny document3), `epochs`, `sg`.

### 5.3 Text preprocessing
Lowercase -> tokenise (`simple_preprocess`) -> remove **stop words** (very common words like "the", "and"). **Lowercase before removing stop words** (gensim's list is lowercase). **Keep negations** ("not", "no", "nor", "never") for sentiment: "not good" must not become "good" (4A uses `STOPWORDS - {"not","no","nor","never"}`).

### 5.4 Document vector and similarity
- **Document vector = average of its word vectors.** Simple and fast; loses word order and negation structure (better: TF-IDF weighting, Doc2Vec, sentence encoders such as BERT).
- **Cosine similarity** = `cos(angle)` between two vectors = `(a . b) / (|a||b|)`; 1 = same direction, 0 = unrelated, -1 = opposite. Preferred over Euclidean distance because it ignores vector length (document length).

### 5.5 Each PS

| PS | What | Settings | Evaluation |
|---|---|---|---|
| **4B** | Word2Vec on tweets | size 100, window 5, min_count 2, 30 epochs | `most_similar`, `similarity`, vector, PCA map. The tweets are **machine-generated random words**, so scores are low and arbitrary (*about 0.26-0.32*): expected |
| **4C** | Word2Vec on `document3.txt` (about 190 words on viruses) | size 50, window 3, **min_count 1**, **100 epochs** | `virus` is near `symptoms`, `covid`. Tiny text: more epochs make every vector identical (at 300) |
| **4D** | **Content-based recommender** | tweets; document vectors; cosine similarity | top-5 most similar tweets with scores *(about 0.43-0.45)* |
| **4A** | **Document classification** (IMDB sentiment, 50,000 reviews) | CBOW vectors averaged -> `Dense(64)`, Dropout, sigmoid, binary cross-entropy | **accuracy *86.3 %***. **Split before training Word2Vec** (no leakage) |

### 5.6 Recommender (4D)
- **Content-based:** represent each item by a vector built from its content; recommend items closest (cosine) to a query or to what the user liked.
- Versus **collaborative filtering**, which uses user-item interactions (ratings, clicks) and needs no content. Content-based avoids the cold-start problem for new items but can only recommend similar things.

### 5.7 Typical questions
- CBOW vs skip-gram? How do you get a document vector? Why cosine and not Euclidean? Why `min_count`? Why keep "not"?
- Limitation of averaging word vectors? Intrinsic evaluation (nearest words, analogies) vs **extrinsic** evaluation (accuracy of a downstream task, as in 4A)?
- Why are your 4B scores random? Why is 4C's text so small and what did you change?

---
## 6. Transfer learning with VGG16 (5A, 5B, 5C)

**One line:** reuse a network trained on a huge dataset (ImageNet) instead of training from scratch; its early layers already detect edges and textures that are useful for almost any image task.

### 6.1 VGG16
- 16 weight layers: **13 convolutional (all 3x3) + 3 dense**; five max-pool stages; about 138 million parameters with the top, about 14.7 million in the convolutional base. Trained on ImageNet (1.4 million images, 1,000 classes).
- `include_top=False` removes the 3 dense layers (the 1000-class classifier), so you attach your own head. The input can then be any size of at least 32x32 (you used 64, 96, 128).
- **Layers go from general to specific:** early layers = edges and textures (reusable), late layers = object parts specific to ImageNet.

### 6.2 Your pipeline
1. **Freeze** the base (`base.trainable = False`): its weights are not updated, so training is fast and there is less risk of over-fitting on small data.
2. **New head:** `GlobalAveragePooling2D` (averages each 512 feature maps to one number: far fewer parameters than Flatten, less over-fitting) -> `Dense(256, relu)` -> `Dropout(0.3)` -> `Dense(classes, softmax)`.
3. **Train the head** (Adam, `1e-3`). This is **feature extraction**.
4. **Fine-tune:** unfreeze the last 4 layers of the base, **re-compile with a very small learning rate (1e-5)**, train a little more. A large rate would destroy the pre-trained weights. Compiling again is required because changing `trainable` only takes effect after compile.
5. **`preprocess_input`** (a Lambda layer): VGG16 was trained with its own scaling (RGB to BGR and subtract the ImageNet mean per channel), **not** `/255`.

### 6.3 Each PS

| PS | Data | Classes | Size | Your result |
|---|---|---|---|---|
| **5B** | Intel Image Classification (buildings, forest, glacier, mountain, sea, street) | 6 | 128 | about *91 % -> 91.5 %* after fine-tuning |
| **5C** | Fashion-MNIST as image folders (skip the stray `train` folder) | 10 | 64 | about 80 % -> 82 % |
| **5A** | **Caltech-101**: 101 object categories (plus a background folder to drop), grayscale images, **imbalanced** (about 31 to 800 images per class), no train/test split | 101 | 96 | about 78 % -> 81 % (the longer `T7` pipeline 85 -> 87 %, top-5 97 %) |

### 6.4 Caltech-101 preparation (Ma'am asked for this one)
Drop `BACKGROUND_Google` (it is not an object), force 3 channels (`color_mode="rgb"`, because some images are grayscale), make an **80/20 split yourself** (`validation_split`, fixed `seed`) because there is no official split, use `preprocess_input`, handle imbalance (class weights or augmentation), and report **top-5 accuracy** and macro-averaged scores, since top-1 alone hides weak small classes.

### 6.5 Typical questions
- Why freeze? Why a small learning rate for fine-tuning? Why `include_top=False`? Why GlobalAveragePooling instead of Flatten?
- Feature extraction vs fine-tuning? Did fine-tuning help? (a little: about +0.5 to +0.8 points in 5B; always report honestly)
- **Is this object detection?** No: it is **whole-image classification** (one label per image). Object detection also predicts bounding boxes (and is evaluated differently); segmentation labels every pixel.
- Why does validation accuracy change when the validation set is shuffled and read twice? Always collect images and labels in one pass.

---
## 7. Datasets at a glance

| Dataset | Used in | Facts |
|---|---|---|
| MNIST | 1A 2B | handwritten digits, 28x28 grey, 60,000 train + 10,000 test, 10 classes |
| Fashion-MNIST | 1B 5C | clothing, 28x28 grey, 60,000 + 10,000, 10 classes (Shirt is the hardest) |
| CIFAR-10 | 1C 2A | 32x32 colour, 10 classes; your CSV has 50,000 rows, split 80/20 |
| Sign-MNIST | 1E 2C 3F | hand-sign letters, 27,455 train + 7,172 test, labels 0-24 (no J, no Z), 25 outputs |
| Credit-card fraud | 3A | 284,807 transactions, 492 fraud (0.17 %), V1-V28 are PCA components |
| ECG | 3B 3E | heartbeats, 140 points, 4,998 rows; label 1 = normal here |
| Energy consumption | 3C | 5,000 half-hourly rows, 5 features, labelled anomalies |
| NASA turbofan | 3D | 100 engines run to failure, 21 sensors; fault = RUL <= 30 |
| IMDB | 4A | 50,000 reviews, positive/negative, balanced |
| Twitter (4B, 4D), document3 (4C) | 4B 4C 4D | random-word tweets; a tiny virus text |
| Intel images | 5B | 6 natural-scene classes, about 12,600 images |
| Caltech-101 | 5A | 101 object classes, imbalanced, no split |

---
## 8. "Tell me about your PS" (your results, one sentence each)

| PS | Say this |
|---|---|
| 1A | FNN with SGD, 11 epochs on MNIST: *94.29 %*; SGD converges slowly but surely. |
| 1B | Same FNN on Fashion-MNIST: *85.55 %*; Shirt is the weakest class (looks like T-shirt and Pullover). |
| 1C | FNN on CIFAR-10: *46.74 %*; flattening loses spatial structure. |
| 1E | FNN on Sign-MNIST, 25 outputs: *72.06 %*; over-fits (train 99 %). |
| 2A | CNN on CIFAR-10: *69.69 %*, 23 points above the FNN. |
| 2B | CNN on MNIST: *98.88 %*. |
| 2C | CNN on Sign-MNIST: *89.26 %*, train 100 %, so it over-fits. |
| 3A | Dense autoencoder on credit cards, mean+3std: *AUC 0.93, recall 0.82, precision 0.22*. |
| 3B | Conv1D autoencoder on ECG: *AUC about 0.98*. |
| 3C | LSTM autoencoder on energy: *AUC 0.69*; the anomalies are random spikes. |
| 3D | LSTM autoencoder on NASA: *AUC 0.92*, precision 0.89, recall 0.46 at the 99th percentile. |
| 3E | Dense autoencoder on ECG: *AUC about 0.99*. |
| 3F | Conv2D autoencoder, letter Y held out: *AUC 0.83, F1 0.18*; weak because Y resembles other letters and the test images differ from the training ones. |
| 4A | CBOW vectors averaged + dense classifier on IMDB: *86.3 %*. |
| 4B / 4C | CBOW on tweets / document3, evaluated by nearest words and similarity. |
| 4D | Content-based recommender: cosine similarity of averaged CBOW vectors, top-5 *(about 0.44)*. |
| 5B / 5C / 5A | VGG16 transfer learning: *about 91.5 %* (Intel), about 82 % (Fashion), about 81 % (Caltech, top-5 97 % in the longer pipeline). |

---
## 9. Formulas sheet
- Dense parameters: `in*out + out`.
- Conv parameters: `(k*k*c_in + 1)*filters`. Output size: valid `(n-k)/s + 1`, same `ceil(n/s)`.
- Softmax: `e^zi / sum(e^zj)`. Cross-entropy: `-log(p_true)`. MSE: `mean((x - x_hat)^2)`.
- Precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, `F1 = 2PR/(P+R)`.
- Gradient descent: `w = w - lr*dL/dw`. Adam: momentum + per-weight scaling, `lr = 1e-3`, `beta1 = 0.9`, `beta2 = 0.999`.
- Cosine similarity: `(a.b)/(|a||b|)`.
- Iterations per epoch: `N / batch`.

---
## 10. Rapid-fire traps (short answers)
- **Why not shuffle test data?** It does not matter for metrics, but for one-pass image-and-label collection use a single pass.
- **Accuracy 99.8 % on fraud, is that good?** No, "always normal" scores that.
- **Train 100 %, validation 89 %?** Over-fitting.
- **Why is my anomaly detector flagging everything?** Test data not scaled like training data, or the threshold is far too low.
- **Why is `RUL` dropped?** The fault label is defined from it (target leakage).
- **Why softmax for multi-class but sigmoid for binary?** Softmax gives probabilities across classes that sum to 1; sigmoid gives one probability for the positive class.
- **Why `Flatten` before `Dense`?** Dense layers take vectors, not 2-D or 3-D maps.
- **Why Adam for CNNs but SGD for 1A and 1B?** The PS specified SGD for 1A/1B; Adam converges faster and is the usual default.
- **Frameworks:** TensorFlow (Google, graph-based, production), **Keras** (high-level API, here running on TensorFlow), PyTorch (Meta, dynamic graphs, popular in research), Theano (the original Python library, no longer developed).
- **ImageDataGenerator vs `image_dataset_from_directory`:** both read class folders (folder name = label); the second is newer and returns a `tf.data` pipeline.
