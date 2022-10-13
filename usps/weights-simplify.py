import json

import numpy as np

from config import *

for i in range(class_num):
    with open(f"{weights_file_prefix}{i}.json", "r") as f:
        json_obj = json.load(f)
        weights = np.round(json_obj["weights"], round_num).astype(float if round_num > 0 else int)
        biases = np.round(json_obj["biases"], round_num).astype(float if round_num > 0 else int)
    with open(f"{weights_file_prefix}{i}.json", "w") as f:
        json.dump({"weights": weights.tolist(), "biases": biases.tolist()}, f)
