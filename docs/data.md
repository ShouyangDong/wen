# 数据集的制作

# 领域知识的基本处理流程
1. 收集领域知识
2. 数据清洗、数据的分块
3. 问答数据的生成

首先我们下载领域知识的文档，将文档中的水印、版权以及无用的小标题去掉，然后根据小的section对文档进行拆分生成一个文本列表，
然后遍历该文本列表，生成对应的数据集。

# 创建一个问答（Q&A）数据集
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
        问题: "您能告诉我 Tensorflow的avg_pool 的源代码吗？", 

        回答： "Tensorflow的avg_pool 的函数调用如下：

            tf.nn.avg_pool(input, ksize, strides, padding, data_format=None, name=None)"
    }, 
    
    
    {
        问题: "介绍Tensorflow的avg_pool的参数。", 
        回答： "Tensorflow的avg_pool函数的参数如下：
        
            - input: Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels] if data_format does not start with "NC" (default), or         [batch_size, num_channels] + input_spatial_shape if data_format starts with "NC". Pooling happens over the spatial dimensions only.
            - ksize：An int or list of ints that has length 1, N or N+2. The size of the window for each dimension of the input tensor.
            - strides: An int or list of ints that has length 1, N or N+2. The stride of the sliding window for each dimension of the input tensor.
            - padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See here for more information.
            - data_format: A string. Specifies the channel dimension. For N=1 it can be either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default) or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
            - name: Optional name for the operation."
    }, 
    ```

3. 根据API的代码，生成对应的代码描述和代码的数据，如：
    ```
    {
        问题: "一个使用MobileNetV2架构在ImageNet数据集上进行训练的预训练图像特征向量模型。它可以用于在新的分类任务上进行特征提取和微调。请写出对应描述的Python代码。"
        
        回答: "input_shape=(224,224,3)\n\n\n# Load an image and preprocess it\nimage = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))\nimage = tf.keras.preprocessing.image.img_to_array(image)\nimage = tf.expand_dims(image, 0)\n\n# Extract feature vector\nfeature_vector = model.predict(image)"
    }
    ```