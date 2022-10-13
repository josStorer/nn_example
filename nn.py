import json
import math

import numpy as np
from tqdm import tqdm


def relu(x):
    return max(0, x)


def derivative_relu(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(target, prediction):
    return ((target - prediction) ** 2).mean()


class nn:
    def __init__(self, input_num, hidden_layer_num, active_func="sigmoid"):
        self.input_num = input_num
        self.hidden_layer_num = hidden_layer_num
        self.output_layer_weight_index = input_num * hidden_layer_num
        self.output_layer_bias_index = hidden_layer_num
        self.weights = [np.random.normal() for _ in range(self.output_layer_weight_index + hidden_layer_num)]
        self.biases = [np.random.normal() for _ in range(self.output_layer_bias_index + 1)]
        self.active_func = active_func

    def active(self, x):
        if (self.active_func == "relu"):
            return relu(x)
        return sigmoid(x)

    def derivative_active(self, x):
        if (self.active_func == "relu"):
            return derivative_relu(x)
        return derivative_sigmoid(x)

    def feedforward(self, x):
        hidden_layers = np.zeros(self.hidden_layer_num)
        hidden_layers_param = np.zeros(self.hidden_layer_num)
        for i in range(self.hidden_layer_num):
            hidden_layers_param[i] = self.biases[i]
            for j in range(self.input_num):
                hidden_layers_param[i] += self.weights[i * self.input_num + j] * x[j]
            hidden_layers[i] = self.active(hidden_layers_param[i])

        output_prediction_param = self.biases[self.output_layer_bias_index]
        for i in range(self.hidden_layer_num):
            output_prediction_param += self.weights[self.output_layer_weight_index + i] * hidden_layers[i]
        output_prediction = self.active(output_prediction_param)
        return output_prediction, output_prediction_param, hidden_layers, hidden_layers_param

    def train(self, data, data_annotation_target, save_file_name="", epochs=1000, learn_rate=0.1):
        loss = math.inf
        for epoch in range(epochs):
            output_predictions = []
            for x, output_target in tqdm(zip(data, data_annotation_target),
                                         f"{save_file_name} - Epoch {epoch} loss {loss:.3f}",
                                         total=len(data)):
                x_flatten = np.array(x).flatten()
                output_prediction, output_prediction_param, hidden_layers, hidden_layers_param \
                    = self.feedforward(x_flatten)
                output_predictions.append(output_prediction)

                # derivative
                d_L_d_pred = -2 * (output_target - output_prediction)

                d_pred_d_w = np.zeros(self.hidden_layer_num)
                d_pred_d_hl = np.zeros(self.hidden_layer_num)
                for i in range(self.hidden_layer_num):
                    d_pred_d_w[i] = hidden_layers[i] * self.derivative_active(output_prediction_param)
                    d_pred_d_hl[i] = self.weights[self.output_layer_weight_index + i] * self.derivative_active(
                        output_prediction_param)
                d_pred_d_b = self.derivative_active(output_prediction_param)

                d_hl_d_w = np.zeros([self.hidden_layer_num, self.input_num])
                d_hl_d_b = np.zeros(self.hidden_layer_num)
                for i in range(self.hidden_layer_num):
                    for j in range(self.input_num):
                        d_hl_d_w[i][j] = x_flatten[j] * self.derivative_active(hidden_layers_param[i])
                    d_hl_d_b[i] = self.derivative_active(hidden_layers_param[i])

                # hidden layer update
                for i in range(self.hidden_layer_num):
                    for j in range(self.input_num):
                        self.weights[i * self.input_num + j] -= learn_rate * d_L_d_pred * d_pred_d_hl[i] * \
                                                                d_hl_d_w[i][j]
                    self.biases[i] -= learn_rate * d_L_d_pred * d_pred_d_hl[i] * d_hl_d_b[i]

                # output prediction update
                for i in range(self.hidden_layer_num):
                    self.weights[self.output_layer_weight_index + i] -= learn_rate * d_L_d_pred * d_pred_d_w[i]
                self.biases[self.output_layer_bias_index] -= learn_rate * d_L_d_pred * d_pred_d_b

            loss = mse_loss(data_annotation_target, output_predictions)
            if (save_file_name != ""):
                with open(save_file_name, "w") as f:
                    json.dump({"weights": self.weights, "biases": self.biases}, f)

    def test(self, data, data_annotation_target, check_fn):
        sum = len(data)
        success = 0
        for x, output_target in tqdm(zip(data, data_annotation_target), "Test", total=sum):
            output_prediction = self.predict(x)
            if (check_fn(output_target, output_prediction)):
                success += 1
        return success / sum

    def read_from_file(self, file_name):
        with open(file_name, "r") as f:
            json_obj = json.load(f)
            self.weights = json_obj["weights"]
            self.biases = json_obj["biases"]

    def predict(self, data):
        data_flatten = np.array(data).flatten()
        result, _, _, _ = self.feedforward(data_flatten)
        return result
