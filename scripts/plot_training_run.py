#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///


import argparse
import json
import collections
import matplotlib.pyplot as plt
import numpy as np


def load_log(file_name: str):
    log_data = json.load(file_name)
    result = collections.defaultdict(list)
    for entry in log_data:
        kind = entry["log"]
        result[kind].append(entry)

    return result


def extract_over_step(data: list[dict], key: str):
    steps = []
    values = []
    for entry in data:
        steps.append(entry["step"])
        values.append(entry[key])
    return steps, values


def main():
    parser = argparse.ArgumentParser(description="Plot training run")
    parser.add_argument("log_file", type=argparse.FileType("r"), help="Log file", default="log.json")
    args = parser.parse_args()

    log_data = load_log(args.log_file)
    steps, losses = extract_over_step(log_data["step"], "loss")
    cmap = plt.get_cmap("tab10")
    plt.plot(steps, losses, c=cmap(0), linewidth=1)
    smoothing = 10
    plt.plot(steps[smoothing:-smoothing], np.convolve(losses, np.ones(2*smoothing+1)/(2*smoothing+1), mode='valid'), c=cmap(0), linewidth=3, label="Training loss")

    steps, losses = extract_over_step(log_data["eval"], "loss")
    plt.plot(steps, losses, c=cmap(1), linewidth=3, label="Validation loss")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
