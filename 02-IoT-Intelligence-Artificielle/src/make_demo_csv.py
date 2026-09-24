"""Build a small demo CSV for the sensor simulation.

UCI HAR windows overlap by 50% (each window shares its last 64 samples with the
next window's first 64 samples), so consecutive windows for one subject can be
stitched back into the original continuous raw recording: keep the first window
whole, then append only the new (last 64) samples of every following window.

We reconstruct the first full activity cycle of test subject 2 this way: it goes
through all 6 activities one after another (STANDING, SITTING, LAYING, WALKING,
WALKING_DOWNSTAIRS, WALKING_UPSTAIRS), which makes a good demo for showing the
simulation react to real transitions between movements.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "UCI HAR Dataset"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "demo.csv"

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
ACTIVITY_NAMES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}
DEMO_SUBJECT = 2
N_WINDOWS = 146  # one full activity cycle for this subject, all 6 activities


def load_ids(path):
    return pd.read_csv(path, header=None, names=["value"]).squeeze("columns")


def load_signal(signal_name):
    path = DATA_DIR / "test" / "Inertial Signals" / f"{signal_name}_test.txt"
    return np.loadtxt(path)


def main():
    subject = load_ids(DATA_DIR / "test" / "subject_test.txt")
    y = load_ids(DATA_DIR / "test" / "y_test.txt")

    rows = subject[subject == DEMO_SUBJECT].index[:N_WINDOWS]
    signals = {name: load_signal(name) for name in CHANNELS}

    stream = {name: [] for name in CHANNELS}
    true_labels = []

    for position, row in enumerate(rows):
        label_name = ACTIVITY_NAMES[y.loc[row]]
        if position == 0:
            for name in CHANNELS:
                stream[name].append(signals[name][row])  # full 128 samples
            true_labels += [label_name] * 128
        else:
            for name in CHANNELS:
                stream[name].append(signals[name][row][64:])  # only the new 64 samples
            true_labels += [label_name] * 64

    data = {name: np.concatenate(stream[name]) for name in CHANNELS}
    df = pd.DataFrame(data)
    df["true_label"] = true_labels

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} with {len(df)} rows ({len(df) / 50:.1f} s at 50 Hz)")
    print(df["true_label"].value_counts())


if __name__ == "__main__":
    main()
