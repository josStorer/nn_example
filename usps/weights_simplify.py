import json

import numpy as np

from usps import config


def run():
    for i in range(config.class_num):
        with open(f"{config.weights_file_prefix}{i}.json", "r") as f:
            json_obj = json.load(f)
            weights = np.round(json_obj["weights"], config.round_num).astype(float if config.round_num > 0 else int)
            biases = np.round(json_obj["biases"], config.round_num).astype(float if config.round_num > 0 else int)
        with open(f"{config.weights_file_prefix}{i}.json", "w") as f:
            json.dump({"weights": weights.tolist(), "biases": biases.tolist()}, f)


if __name__ == "__main__":
    run()
