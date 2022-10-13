import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from nn import nn
import sys

class_num = 10
img_size = (16, 16)
hidden_layer_num = 1

network = [nn(img_size[0] * img_size[1], 1, "sigmoid") for _ in range(class_num)]
for i in range(class_num):
    network[i].read_from_file(f"usps-weights-{i}.json")


def predict(data):
    results = np.zeros(class_num)
    for i in range(class_num):
        results[i] = network[i].predict(data)
    return results.argmax()


img_name = "img.jpg"
if (len(sys.argv) > 1):
    img_name = sys.argv[1]


def read_img_and_predict():
    img = cv2.imread(img_name)
    img = cv2.resize(img, dsize=(img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
    img = 1 - (np.array(img) / 255)[:, :, 0]
    plt.imshow(img, cmap='gray')
    plt.title(f"Predict: {predict(img)}")
    plt.show()


class img_modified(LoggingEventHandler):
    def on_modified(self, event):
        super(LoggingEventHandler, self).on_modified(event)
        if (img_name in event.src_path):
            try:
                read_img_and_predict()
            except:
                pass


read_img_and_predict()
observer = Observer()
observer.schedule(img_modified(), '.')
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
