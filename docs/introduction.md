# 介绍

需求描述
打造 特定领域知识(Domain-specific Knowledge) 问答 系统，具体需求有：

* 通过自然语言问答的形式，和用户交互，同时支持中文和英文。
* 理解用户不同形式的问题，找到与之匹配的答案。可以* 对答案进行二次处理，比如将关联的多个知识点进行去重、汇总等。
* 支持上下文。有些问题可能比较复杂，或者原始知识不能覆盖，需要从历史会话中提取信息。
* 准确。不要出现似是而非或无意义的回答。


本文提出一种由大模型+search的方式，充分利用大模型的思维链的推理能力，将问题的背景文档进行归纳总结。高效、准确的找出其对应的答案。整个文档的介绍将从以下几个方面介绍整个知识问答系统：

* [数据集的制作](./llm.md)
* [模型精调](./data.md)
* [模型评测](./evaluate.md)
* [数据库的构建](./database.md)
* 模型web gui的展示