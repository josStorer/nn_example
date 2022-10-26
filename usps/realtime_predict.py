import pathlib
import time

import numpy as np
import cv2
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import threading

from nn import nn
from usps import config
import sys


def predict(data):
    global networks
    results = np.zeros(config.class_num)
    for i in range(config.class_num):
        results[i] = networks[i].predict(data)
    return results.argmax()


def get_current_folder():
    return pathlib.Path(__file__).parent.resolve().__str__()


def read_img_and_predict():
    global img
    global img_name
    global pycharm_mode
    img = cv2.imread(get_current_folder() + "/" + img_name)
    img = cv2.resize(img, dsize=(config.img_size[0], config.img_size[1]), interpolation=cv2.INTER_CUBIC)
    img = 1 - (np.array(img) / 255)[:, :, 0]
    if pycharm_mode:
        import matplotlib.pyplot as plt
        plt.imshow(img, cmap="gray")
        plt.title(f"Predict: {predict(img)}")
        plt.show()


def img_show_thread():
    global img
    while True:
        predict_name = f"Predict: {predict(img)}"
        display_img = cv2.resize(img, dsize=(config.preview_window_size, config.preview_window_size))
        cv2.putText(display_img, predict_name, (0, config.preview_window_size - config.preview_font_thickness),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.preview_font_scale, (255, 255, 255), config.preview_font_thickness)
        cv2.imshow("", display_img)
        cv2.resizeWindow("", (config.preview_window_size, config.preview_window_size))
        cv2.waitKey(30)
        if not observer.is_alive():
            return


class img_modified(LoggingEventHandler):
    global img_name

    def on_modified(self, event):
        super(LoggingEventHandler, self).on_modified(event)
        if img_name in event.src_path:
            try:
                read_img_and_predict()
            except:
                pass


def run(file_name="", use_pycharm=False):
    global pycharm_mode
    global img_name
    global networks
    global observer
    pycharm_mode = use_pycharm
    img_name = file_name if file_name != "" else config.default_img_name
    networks = [nn(config.img_size[0] * config.img_size[1], 1, "sigmoid") for _ in range(config.class_num)]
    for i in range(config.class_num):
        networks[i].read_from_file(f"{config.weights_file_prefix}{i}.json")

    read_img_and_predict()
    if not pycharm_mode:
        threading.Thread(target=img_show_thread).start()

    observer = Observer()
    observer.schedule(img_modified(), get_current_folder())
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run("" if len(sys.argv) == 1 else sys.argv[1],
        True)
