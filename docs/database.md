# 数据库处理

首先我们将文档拆成若干个小章节， 比如

```
TensorBoard：TensorFlow 的可视化工具包


TensorBoard 提供机器学习实验所需的可视化功能和工具：

 * 跟踪和可视化损失及准确率等指标

 * 可视化模型图（操作和层）

 * 查看权重、偏差或其他张量随时间变化的直方图

 * 将嵌入投射到较低的维度空间

 * 显示图片、文字和音频数据

 * 剖析 TensorFlow 程序

 * 以及更多功能
```

另外也可以将文档API进行分开， 比如：

```
tf.nn.avg_pool

Performs the avg pooling on the input.

tf.nn.avg_pool(
    input, ksize, strides, padding, data_format=None, name=None
)

Each entry in output is the mean of the corresponding size ksize window in value.

Args

- input: Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels] if data_format does not start with "NC" (default), or [batch_size, num_channels] + input_spatial_shape if data_format starts with "NC". Pooling happens over the spatial dimensions only.
- ksize: An int or list of ints that has length 1, N or N+2. The size of the window for each dimension of the input tensor.
strides	An int or list of ints that has length 1, N or N+2. The stride of the sliding window for each dimension of the input tensor.
- padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See here for more information.
- data_format: A string. Specifies the channel dimension. For N=1 it can be either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default) or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
- name:	Optional name for the operation.

Returns
A Tensor of format specified by data_format. The average pooled output tensor.
```
通过``ingest.py`中的函数对每个的小章节或者API介绍进行编码，并保存。