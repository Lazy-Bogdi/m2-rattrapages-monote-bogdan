"""Experiment: is a smaller CNN (half the filters of the final model) good enough?

Uses the same train/validation split and normalization already saved by
02_preprocessing.ipynb (not recomputed), and only looks at validation accuracy
here, the test set stays untouched until the model is frozen, as decided for
the final model in 03_training.ipynb.

Not used for the final deliverable: this script only prints results, it does
not save a model. Kept as a research trace (see notes.md for the outcome).
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "UCI HAR Dataset"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

with open(MODELS_DIR / "normalization.json") as f:
    normalization = json.load(f)

CHANNELS = normalization["channels"]
mean = np.array(normalization["mean"])
std = np.array(normalization["std"])
validation_subjects = normalization["validation_subjects"]


def load_ids(path):
    return pd.read_csv(path, header=None, names=["value"]).squeeze("columns")


def load_signal(split, signal_name):
    path = DATA_DIR / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    return np.loadtxt(path)


def load_windows(split):
    return np.stack([load_signal(split, name) for name in CHANNELS], axis=-1)


def normalize(X):
    return (X - mean) / std


X_train_full = load_windows("train")
subject_train_full = load_ids(DATA_DIR / "train" / "subject_train.txt")
y_train_full = (load_ids(DATA_DIR / "train" / "y_train.txt") - 1).to_numpy()

val_mask = subject_train_full.isin(validation_subjects).to_numpy()
fit_mask = ~val_mask

X_fit = normalize(X_train_full[fit_mask])
y_fit = y_train_full[fit_mask]
X_val = normalize(X_train_full[val_mask])
y_val = y_train_full[val_mask]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, len(CHANNELS))),
    tf.keras.layers.Conv1D(8, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(16, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

n_params = model.count_params()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(
    X_fit, y_fit, validation_data=(X_val, y_val),
    epochs=100, batch_size=64, callbacks=[early_stopping], verbose=0,
)

best_val_accuracy = max(history.history["val_accuracy"])
print(f"params: {n_params}")
print(f"epochs run: {len(history.history['loss'])}")
print(f"best val_accuracy: {best_val_accuracy:.4f}")
