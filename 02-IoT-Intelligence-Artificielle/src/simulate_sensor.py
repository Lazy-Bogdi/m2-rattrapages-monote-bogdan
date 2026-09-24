"""Simulate a sensor streaming into the trained activity recognition model.

Reads a CSV file line by line, as if each line just arrived from an accelerometer
and gyroscope (like it would on an ESP32), waits between lines to mimic real time,
keeps the last `window_length` samples in a sliding buffer, and runs inference
every `--step` samples once the buffer is full.

Example:
    python simulate_sensor.py --speed 10
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate an IoT sensor stream and predict activities.")
    parser.add_argument("--file", type=Path, default=DATA_DIR / "demo.csv", help="CSV file to stream from.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "model_int8.tflite", help=".tflite model to use.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (10 = 10x faster than real time).")
    parser.add_argument("--step", type=int, default=16, help="Run inference every N new samples once the buffer is full.")
    parser.add_argument("--uncertain-threshold", type=float, default=0.6, help="Below this confidence, print 'uncertain' instead of the predicted label.")
    return parser.parse_args()


def load_metadata():
    with open(MODELS_DIR / "normalization.json") as f:
        normalization = json.load(f)
    with open(MODELS_DIR / "labels.json") as f:
        labels_map = json.load(f)
    return normalization, labels_map


def make_predictor(model_path):
    """Return a predict(window) -> (label_index, confidence) function for a .tflite model.

    Handles both a float32 model (normalized floats in and out) and a fully
    int8-quantized model (which needs its inputs quantized and its output
    dequantized using the scale/zero_point stored in the model itself).
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    is_quantized = input_details["dtype"] == np.int8

    def predict(window):
        sample = window[np.newaxis, :, :].astype("float32")
        if is_quantized:
            scale, zero_point = input_details["quantization"]
            sample = np.round(sample / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] == np.int8:
            scale, zero_point = output_details["quantization"]
            output = (output.astype("float32") - zero_point) * scale
        label_index = int(np.argmax(output))
        confidence = float(output[label_index])
        return label_index, confidence

    return predict


def main():
    args = parse_args()
    normalization, labels_map = load_metadata()
    channels = normalization["channels"]
    mean = np.array(normalization["mean"])
    std = np.array(normalization["std"])
    window_length = normalization["window_length"]
    sampling_rate_hz = normalization["sampling_rate_hz"]

    predict = make_predictor(args.model)
    buffer = deque(maxlen=window_length)

    df = pd.read_csv(args.file)
    has_ground_truth = "true_label" in df.columns

    sample_period = (1.0 / sampling_rate_hz) / args.speed
    since_last_prediction = 0

    print(f"streaming {len(df)} samples from {args.file.name} using {args.model.name} "
          f"(speed x{args.speed}, window={window_length} samples)\n")

    for i, row in df.iterrows():
        buffer.append(row[channels].to_numpy(dtype="float64"))
        since_last_prediction += 1
        time.sleep(sample_period)

        buffer_full = len(buffer) == window_length
        time_to_predict = buffer_full and (since_last_prediction >= args.step or i == window_length - 1)
        if not time_to_predict:
            continue

        since_last_prediction = 0
        window = (np.array(buffer) - mean) / std
        label_index, confidence = predict(window)
        label = labels_map[str(label_index)]
        elapsed_s = i / sampling_rate_hz

        if confidence < args.uncertain_threshold:
            prediction_text = f"uncertain (closest: {label}, {confidence:.0%})"
        else:
            prediction_text = f"{label} ({confidence:.0%})"

        line = f"t={elapsed_s:6.2f}s  predicted: {prediction_text}"
        if has_ground_truth:
            line += f"   true: {row['true_label']}"
        print(line)


if __name__ == "__main__":
    main()
