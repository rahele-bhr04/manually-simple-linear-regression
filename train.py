import random
import numpy as np
import dataset as ds
from dataset import X_train, y_train

# y_true = wx + b
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# learning rate
α = 0.01

#update w and b, minibatch
batch_size = 25

for epoch in range(1000):

    # Shuffle the data
    indices = np.random.permutation(len(ds.X_train))
    X_train_shuffled = ds.X_train[indices]
    y_train_shuffled = ds.y_train[indices]

    # Go through mini-batches
    for i in range(0, len(X_train_shuffled), batch_size):
        X_batch = X_train_shuffled[i: i+ batch_size]
        y_batch = y_train_shuffled[i: i+ batch_size]

        # Predictions
        y_pred = w * X_batch + b

        # Compute gradients
        dw = 2 * np.mean(X_batch * (y_pred - y_batch))
        db = 2 * np.mean(y_pred - y_batch)

        # Update parameters
        w -= α * dw
        b -= α * db