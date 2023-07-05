# 数据集的制作

# 领域知识的基本处理流程
1. 收集领域知识
2. 数据清洗、数据的分块
3. 数据的存储


# 创建一个合成问答（Q&A）数据集
我们主要使用chatgpt和规则算法的方式， 根据给定的文本生成对应的问答数据。具体的流程如下：

* 加载数据，生成对应的文本信息
* 根据对应的文本生成问答数据

1. 用户可以使用`data_process.py`, 采用chatgpt根据背景文档，生成若干个问答数据， 然后根据问答数据进行筛选。比如对Tensorboard的介绍的问答为：
```
    问题：TensorBoard是什么？
    回答：TensorBoard是TensorFlow的可视化工具包，提供机器学习实验所需的可视化功能和工具。

    问题：TensorBoard可以跟踪和可视化哪些指标？
    回答：TensorBoard可以跟踪和可视化损失、准确率等指标。

    问题：TensorBoard可以用来查看哪些内容的直方图？
    回答：TensorBoard可以用来查看权重、偏差或其他张量随时间变化的直方图。

    问题：TensorBoard有哪些功能？
    回答：TensorBoard可以可视化模型图、将嵌入投射到较低的维度空间、显示图片、文字和音频数据、剖析TensorFlow程序等，并提供更多功能。
```


2. 根据API接口（结构比较固定），我们这里采用规则算法，生成对应的QA数据，如：

    ```
    {
        "input": "", 
        "instruction": "您能告诉我 Tensorflow的avg_pool 的源代码吗？", 

        "output": "Tensorflow的avg_pool 的函数调用如下：\n

        tf.nn.avg_pool(input, ksize, strides, padding, data_format=None, name=None)"
    }, 
    
    
    {
        "input": "", "instruction": "介绍Tensorflow的avg_pool的参数。", 
        "output": "Tensorflow的avg_pool函数的参数如下：\n

        input: 等级为 N+2 的张量，如果 data_format 不以“NC”开头（默认），则形状为 [batch_size] + input_spatial_shape + [num_channels]，或者如果 data_format 以“NC”开头，则为 [batch_size, num_channels] + input_spatial_shape。 池化仅发生在空间维度上。

        - ksize: 长度为 1、N 或 N+2 的整数或整数列表。 输入张量每个维度的窗口大小。

        - strides：长度为 1、N 或 N+2 的整数或整数列表。 输入张量每个维度的滑动窗口的步幅。

        - padding:一个字符串，“VALID”或“SAME”。 填充算法。 浏览此处获取更多信息。

        - data_format: 一个字符串。 指定通道尺寸。 对于 N=1，它可以是“NWC”（默认）或“NCW”，对于 N=2，它可以是“NHWC”（默认）或“NCHW”，对于 N=3，它可以是“NDHWC”（默认）或 “NCDHW”。

        - name: 操作的可选名称."
    }, 
    ```