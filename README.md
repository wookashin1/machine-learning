# Machine Learning

Homework for the Machine Learning course at the School of Computing (RAF),
Union University. Most of the models are written by hand, on top of low-level
TensorFlow ops or plain NumPy, so that the training and prediction steps are
visible instead of hidden behind `model.fit()`. The last project is a CNN that
learns to steer a car in the Udacity simulator by imitating a human driver.

| # | Task | Implementation | Result |
|---|------|----------------|--------|
| hw1/2a | Polynomial regression, degree 1–6 | TF variables + `GradientTape`, Adam | loss compared across degrees |
| hw1/2b | Degree-4 regression with L2 regularization | same, λ from 0 to 100 | effect of λ on weights and loss |
| hw1/3a–3c | k-Nearest Neighbors | TF tensor ops, no library model | Iris: 0.70–0.83 with 2 features, 0.90–0.97 with all 4 |
| hw1/4 | Multinomial Naive Bayes for text | NumPy, Laplace smoothing | Disaster Tweets: 0.80 accuracy |
| hw2/1 | Decision tree | scikit-learn | Titanic: ~0.75–0.78 validation accuracy |
| hw2/2 | Neural network with one hidden layer | NumPy, backpropagation by hand | Titanic: 81% validation accuracy |
| hw2/3 | Behavioral cloning (self-driving) | Keras CNN (NVIDIA architecture) | steers the car in the Udacity simulator |

## Homework 1

**Polynomial regression (`2a`, `2b`).** Predicts water temperature from
salinity on the first 700 clean rows of the CalCOFI `bottle.csv` dataset. The
features are powers of the normalized input. The weights are TensorFlow
variables, with gradients computed by `GradientTape` and updated by Adam. `2a` fits polynomials of degree 1 to
6 and compares the final loss; `2b` fixes the degree at 4 and adds an L2
penalty, sweeping λ over 0, 0.001, 0.01, 0.1, 1, 10 and 100 to show how the
weights shrink towards zero as λ grows.

**k-NN (`3a`, `3b`, `3c`).** An unweighted k-NN classifier on Iris, built from
TensorFlow tensor ops: Euclidean distances, `top_k` for the nearest
neighbours, and a one-hot vote.
- `3a`: two features (sepal length and width), k = 3: 0.77 test accuracy, with
  the decision regions plotted.
- `3b`: the same two features, k from 1 to 15: 0.70–0.83.
- `3c`: all four features, k from 1 to 15: 0.90–0.97. The petal measurements
  separate the classes much better than the sepal ones.

**Naive Bayes text classification (`4`).** Classifies tweets as being about a
real disaster or not (Kaggle *NLP with Disaster Tweets*). Everything is
implemented by hand:
- preprocessing: lowercasing; removing URLs, mentions, hashtags and
  non-letters; removing stop words;
- a vocabulary of the 10,000 most frequent words and bag-of-words vectors;
- Multinomial Naive Bayes with Laplace smoothing, with prediction in log space
  to avoid underflow.

It reaches **0.80** accuracy on an 80/20 split. The notebook also ranks words
by the ratio of their counts in disaster and non-disaster tweets. The most
"disaster" words are *train, fires, fatal, injured*; the least are *body,
love, night*. That ratio says more about a word than raw frequency does.

## Homework 2

**Titanic: decision tree (`1.py`) and a hand-written neural network
(`2.py`).** Both use the same preprocessing: median imputation for age and
fare, the most frequent value for the port of embarkation, and integer
encoding of categorical columns. `1.py` trains a scikit-learn decision tree
and prints its rules. `2.py` implements a neural network with one hidden
layer (10 sigmoid units) in NumPy, including the forward pass, binary
cross-entropy and backpropagation, and trains it with full-batch gradient
descent for 1,000 epochs.

**Behavioral cloning (`3. zadatak ML/`).** A convolutional network that
predicts the steering angle from a single front-camera image. It is trained
on 4,180 recorded driving frames, each with images from the center, left and
right cameras.
- **Model** (`model.py`): the NVIDIA end-to-end driving architecture. It has
  normalization, five convolutional layers (24-36-48-64-64), dropout and
  fully connected layers 100-50-10-1, and is trained with MSE loss and Adam.
- **Preprocessing** (`utils.py`): crop the sky and the hood, resize to
  66×200, convert to YUV.
- **Augmentation**, applied to 60% of training samples:
  - pick a random camera and correct the angle by ±0.2;
  - random horizontal flip, with the angle negated;
  - random translation, with the angle adjusted;
  - random shadow and random brightness.
- **Driving** (`drive.py`): a Flask-SocketIO server receives telemetry from the
  simulator, predicts the angle, and sets the throttle to slow down in
  curves.

## Running it

The datasets for homework 1 are not in the repository because of their size
(`bottle.csv` alone is 250 MB). Download them and put them next to the
notebooks in `hw1/`:
- `bottle.csv`: [CalCOFI](https://www.kaggle.com/datasets/sohier/calcofi)
- `iris.csv`: [Iris](https://www.kaggle.com/datasets/uciml/iris)
- `disaster-tweets.csv`: the `train.csv` file from [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

```bash
# Homework 2: Titanic (data included)
python hw2/1.py
python hw2/2.py

# Homework 2: self-driving car
cd "hw2/3. zadatak ML"
pip install -r requirements.txt            # TensorFlow 2.10 / Keras 2.10
python model.py -d data                    # trains, saves model/model-XXX.h5
python drive.py -m model/model-XXX.h5      # then start the simulator in autonomous mode
```

For the self-driving project you need the
[Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim).
Record your own driving data in training mode: `data/driving_log.csv` and
`data/IMG/`. The folder also has a `.devcontainer` configuration with the
`tensorflow/tensorflow:2.10.0-gpu` image.
