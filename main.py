import numpy as np

from nn import nn

# dataset
data = np.array([
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],

    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],

    [[1, 0, 1],
     [1, 1, 1],
     [0, 0, 1]],

    [[1, 1, 1],
     [0, 0, 1],
     [0, 0, 1]]
])

normalization = 10

data_annotation_target = np.array([
    0, 1, 4, 7
]) / normalization

network = nn(3 * 3, 9)
network.train(data, data_annotation_target, "weights.json")

network.read_from_file("weights.json")
for i, _ in enumerate(data):
    predict = round(network.predict(data[i]) * normalization)
    print(f"predict: {predict}, result: {predict == data_annotation_target[i] * normalization}")
