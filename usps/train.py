import sys

import numpy as np
from sklearn.datasets import fetch_openml

from nn import nn
from config import *

usps = fetch_openml("usps", version=2, as_frame=False)
data = usps["data"][:train_index]
data_annotation_target = np.array(usps["target"][:train_index]).astype(float)

network = [nn(img_size[0] * img_size[1], hidden_layer_num, "sigmoid") for _ in range(class_num)]
for i in range(class_num):
    if read:
        network[i].read_from_file(f"{weights_file_prefix}{i}.json")
    if (train or (len(sys.argv) > 1 and sys.argv[1] == "train")):
        current_target = np.zeros(len(data_annotation_target))
        for j, _ in enumerate(data_annotation_target):
            if data_annotation_target[j] == i + 1:
                current_target[j] = 1
        network[i].train(data, current_target, f"{weights_file_prefix}{i}.json", epochs)


def predict(data):
    results = np.zeros(class_num)
    for i in range(class_num):
        results[i] = network[i].predict(data)
    return results.argmax()


test_data = usps["data"][test_index_start:test_index_end]
test_annotation_target = np.array(usps["target"][test_index_start:test_index_end]).astype(float)
total = len(test_data)
success = 0
for i in range(total):
    result = predict(test_data[i])
    if result == test_annotation_target[i] - 1:
        success += 1
print(f"{success / total * 100:.2f}%")
