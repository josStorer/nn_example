# **[简体中文](./README_ZH.md) | English**

This repository is related to the [video about principles of neural networks](https://www.bilibili.com/video/BV1fd4y1y7xS/?p=2), which demonstrates how to write a simple neural network and implement handwritten digit recognition without using a framework.

The neural network has a single hidden layer, and one output, you can change some arguments in the [config file](./usps/config.py) to adjust the settings for model training/simplification/preview detection

## Start

```shell
git clone https://github.com/josStorer/nn_example.git --depth=1
cd nn_example
pip install -r requirements.txt
python cli.py -h
````

## Command example

```shell
python cli.py -h                 # get help
python cli.py -test              # test accuracy
python cli.py -m                 # mini example test
python cli.py -train             # start training the model
python cli.py -s                 # simplify trained weights
python cli.py -p                 # start the real-time preview detection, open img.jpg in usps folder with a paint software, then edit and save, the result will be automatically refreshed
python cli.py -p -pycharm        # pycharm preview mode, automatically refresh with SciView. When "right-click" running ./usps/realtime_predict.py in pycharm, this mode is True
python cli.py -p -pf [filename]  # specify the image filename of the real-time preview
````

***

### Notes

The number of classes can be modified in config.py, and the data source and data annotation can be modified in train.py

The data label value starts from 1, indicating category 1, 2, 3, ... 99, 100 ...

Finally, networks with the number of [class_num] are generated, each of which is used to predict whether the given data is of category 1, 2, 3, ... 99, 100 ...

When implementing classification, the network index with the prediction result closest to 1 is the currently judged category. The implementation code is as follows:

````python
results = np.zeros(config.class_num)
for i in range(config.class_num):
    results[i] = network[i].predict(data)
return results.argmax() # This result starts from 0, so you have to add 1 to get the above category labelled value
````

***
reference: https://victorzhou.com/blog/intro-to-neural-networks/