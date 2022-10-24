# **简体中文 | [English](./README.md)**

这是一个与[神经网络原理讲解视频](), 相配套的项目, 演示在不使用框架的情况下, 编写简易的神经网络, 并实现手写数字识别

该神经网络具有单层隐藏层, 单输出, 你可以在[配置文件](./usps/config.py)中调整一些参数, 以调整模型训练/简化/预览检测的设置

## 使用方法

```shell
git clone https://github.com/josStorer/nn_example.git --depth=1
cd nn_example
pip install -r requirements.txt
python cli.py -h
```

## 命令示例

```shell
python cli.py -h              # 获取帮助
python cli.py -test           # 测试准确率
python cli.py -m              # 迷你示例测试
python cli.py -train          # 开始训练模型
python cli.py -s              # 简化训练完毕后的权重
python cli.py -p              # 启动实时预览检测, 使用画图软件打开usps目录下的img.jpg图片, 编辑并保存, 结果将自动刷新
python cli.py -p -pycharm     # pycharm预览模式, 配合SciView自动刷新, 使用pycharm在./usps/realtime_predict.py文件下右键运行自动使用此模式
python cli.py -p -pf [文件名]  # 指定实时预览的图像文件名
```

***

### 附注

config.py中可修改类数量, train.py中可修改数据源和数据标注

数据标注值从1开始, 指示类别, 1, 2, 3, ... 99, 100 ...

最终生成class_num个网络, 各用于判断给定数据是否是类别1, 2, 3, ... 99, 100 ...

实现分类时, 预测结果最接近1的网络索引即是当前判断的类别, 实现代码如下:

```python
results = np.zeros(config.class_num)
for i in range(config.class_num):
    results[i] = network[i].predict(data)
return results.argmax()  # 这个结果从0开始, 因此需要加1才是上述的类别标注值
```
