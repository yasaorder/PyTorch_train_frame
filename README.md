# PyTorch_train_frame
PyTorch深度学习训练框架，包括训练框架、测试数据集、评估指标（待做）、可视化、损失函数

- 留一个大佬的Pytorch常用代码段的文章：
https://zhuanlan.zhihu.com/p/447351423
## Dataset.py
读取数据集， 含数据增强， 旋转、反转、随机饱和度等
## Frame.py
训练框架， 功能包含早停、输出损失曲线、保存最优权重、输出日志文件，同时含设置随机种子的函数，保证结果可复现
## check_data.py
检查数据集是否正确， 依次读取图像、json标签、gt mask
## modules.py
包含各种损失函数和注意力机制模块
## visdual.py
可视化模块， 包含可视化图像、可视化预测掩膜、可视化损失函数
终端输入
```python
 tensorboard --logdir 'log/visdual'
```
 ————这里为存放日志文件的文件夹
## mytrain.py
训练网络的代码， 可根据需要修改， 包含train函数和test函数，分别进行训练和测试
## experiment.py
训练作业文件， 方便在无人情况下进行多次不同参数训练， 包括训练、检查数据集、测试
