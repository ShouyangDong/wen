# 领域适应

**领域适应**的目标是使文本嵌入模型适应您的特定文本域，而不需要标记训练数据。

领域适应仍然是一个活跃的研究领域，目前还没有完美的解决方案。在[TSDAE](https://arxiv.org/abs/2104.06979) 和 [GPL](https://arxiv.org/abs/2112.07577)文章中，作者评估了几种如何使文本嵌入模型适应您的特定领域的方法。

# 领域适应与无监督学习

存在无监督文本嵌入学习的方法，但是它们通常表现相当糟糕：它们并不真正能够学习特定领域的概念。

更好的方法是领域适应：在这里，您拥有来自特定领域的未标记语料库以及现有的标记语料库。您可以在这里找到许多合适的标记训练数据集：[embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)

# 自适应预训练

当使用自适应预训练时，您首先使用例如在目标语料库上进行预训练。掩码语言建模或 TSDAE，然后对现有训练数据集进行微调（请参阅[嵌入训练数据](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)）。

![Adaptive Pre-Training](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/adaptive_pre-training.png) 

在论文中 [TSDAE](https://arxiv.org/abs/2104.06979) 作者评估了 4 个领域特定句子嵌入任务的几种领域适应方法：

| Approach | AskUbuntu | CQADupStack | Twitter | SciDocs | Avg |
| -------- | :-------: | :---------: | :-----: | :-----: | :---: |
| Zero-Shot Model | 54.5 | 12.9 | 72.2 | 69.4 | 52.3 |
| TSDAE | 59.4 | **14.4** | **74.5** | **77.6** | **56.5** |
| MLM | **60.6** | 14.3 | 71.8 |  76.9 | 55.9 |
| CT | 56.4 | 13.4 | 72.4 |  69.7 | 53.0 |
| SimCSE | 56.2 | 13.1 | 71.4 | 68.9 | 52.4 |


正如我们所看到的，当您首先对特定语料库进行预训练，然后对提供的标记训练数据进行微调时，性能最多可以提高 8 个点。

自适应预训练的一大缺点是计算开销较高，因为您必须首先在语料库上运行预训练，然后在标记的训练数据集上进行监督学习。带标签的训练数据集可能非常大（例如，“all-*-v1”模型已在超过 10 亿个训练对上进行了训练）。

目前`Wen`中采用了[数据集的制作](./data.md)中利用大模型和模板规则的方法生成QA数据，然后采用[大模型训练细节](./llm.md)中先采用预训练后精调的方式，通过实验结构观察该方法确实较大的提升了整个大模型问答的性能。

### Citation

 [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979)
```bibtex 
@inproceedings{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and Gurevych, Iryna", 
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "671--688",
    url = "https://arxiv.org/abs/2104.06979",
}
```

[GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://arxiv.org/abs/2112.07577):
```bibtex  
@inproceedings{wang-2021-GPL,
    title = "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval",
    author = "Wang, Kexin and Thakur, Nandan and Reimers, Nils and Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2112.07577",
    month = "12",
    year = "2021",
    url = "https://arxiv.org/abs/2112.07577",
}
```