import sys
import multiprocessing as mp
import time

import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from nn import nn
from usps import config

usps = fetch_openml("usps", version=2, as_frame=False)
data = usps["data"][:config.train_index]


def init_nn():
    return nn(config.img_size[0] * config.img_size[1], config.hidden_layer_num, "sigmoid")


def train_process(index, current_target):
    network = init_nn()
    if config.read:
        network.read_from_file(f"{config.weights_file_prefix}{index}.json")
    network.train(data, current_target, f"{config.weights_file_prefix}{index}.json", config.epochs)


def run(is_train):
    def train():
        data_annotation_target = np.array(usps["target"][:config.train_index]).astype(float)

        pool_size = config.process_num if config.process_num > 0 else \
            min(
                max(1, mp.cpu_count() - 2),
                config.class_num
            )
        pool = mp.Pool(pool_size)
        pool_results = []

        for i in range(config.class_num):
            current_target = np.zeros(len(data_annotation_target))
            for j, _ in enumerate(data_annotation_target):
                if data_annotation_target[j] == i + 1:
                    current_target[j] = 1
            pool_results.append(pool.apply_async(train_process, args=[i, current_target]))

        try:
            while not all([result.ready() for result in pool_results]):
                time.sleep(1)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()

    def test():
        def predict(data):
            results = np.zeros(config.class_num)
            for i in range(config.class_num):
                results[i] = networks[i].predict(data)
            return results.argmax()

        networks = [init_nn() for _ in range(config.class_num)]
        for i in range(config.class_num):
            networks[i].read_from_file(f"{config.weights_file_prefix}{i}.json")
        test_data = usps["data"][config.test_index_start:config.test_index_end]
        test_annotation_target = np.array(usps["target"][config.test_index_start:config.test_index_end]).astype(float)
        total = len(test_data)
        success = 0
        for i in tqdm(range(total), "Test accuracy", total=total):
            result = predict(test_data[i])
            if result == test_annotation_target[i] - 1:
                success += 1
        print(f"{success / total * 100:.2f}%")

    if is_train:
        train()

    test()


if __name__ == "__main__":
    run(config.train or (len(sys.argv) > 1 and sys.argv[1] == "train"))
