# 训练细节

!重要提示！

* 因为PEFT库变动频繁，该代码仅适用于特定的peft版本，请从源码安装commit id为13e53fc的Peft。如果使用其他版本的PEFT， 不能保证模型可以正常训练。
* Pre-training Stage 1中模型训练收敛速度较慢，我们不推荐使用，目录中也没有提供相应的脚本。
* 运行前确保拉仓库的最新代码： git pull

预训练和精调可以参考对应的开源大模型的训练步骤， 我们这以[中文alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)为例。

# 训练步骤

以下介绍的是Pre-training Stage2 开始训练的步骤，训练脚本：[scripts/training/run_clm_pt_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_clm_pt_with_peft.py)

进入项目的`scripts/training`目录，运行`bash run_pt.sh`进行指令精调， 默认使用单卡。运行前用户应先修改脚本并指定相关参数，脚本中的相关参数值仅供调试参考。 `run_pt.sh`的内容如下：

```
########参数设置########
lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/llama/dir
chinese_tokenizer_path=path/to/chinese/llama/tokenizer/dir
dataset_dir=path/to/pt/data/dir
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=output_dir

deepspeed_config_file=ds_zero2_no_offload.json

########启动命令########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```
以下是脚本支持的训练模式，请根据相应的情况传入`model_name_or_path`和`tokenizer_name_or_path`参数。不支持在表格中的模式，如要修改请自行debug。

用途	model_name_or_path	tokenizer_name_or_path	最终模型词表大小
基于原版LLaMA继续训练（词表不变）	原版HF格式的LLaMA	原版LLaMA的tokenizer（32000）	32000
基于原版LLaMA训练中文LLaMA	原版HF格式的LLaMA	中文LLaMA的tokenizer（49953）	49953
基于中文LLaMA继续预训练	HF格式的中文LLaMA/LLaMA-Plus	中文LLaMA的tokenizer（49953）	49953
基于中文Alpaca继续预训练	HF格式的中文Alpaca/Alpaca-Plus	中文Alpaca的tokenizer（49954）	49954


其他部分参数的解释如下：

* --dataset_dir: 预训练数据的目录，可包含多个以txt结尾的纯文本文件
* --data_cache_dir: 指定一个存放数据缓存文件的目录

这里列出的其他训练相关超参数，尤其是学习率以及和total batch size大小相关参数仅供参考。请在实际使用时根据数据情况以及硬件条件进行配置。

# 使用多机多卡训练

请参考以下启动方式：

```
torchrun \
  --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node} 
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    ...
```

训练后文件整理
训练后的LoRA权重和配置存放于${output_dir}/pt_lora_model，可用于后续的合并流程。

（以下文件整理步骤已被整合到训练脚本，无需执行，此处仅供留存参考，并将在之后的更新中删除）

1. 创建一个文件夹${lora_model}，用于存放LoRA模型

2. 将${output_dir}中训练好的pytorch_model.bin移动到${lora_model}，并命名为adapter_model.bin

    ```mv ${output_dir}/pytorch_model.bin ${lora_model}/adapter_model.bin```

3. 将Chinese-LLaMA-LoRA（7B、13B、是否是Plus均可）中的tokenizer相关文件复制到${lora_model}

    ```cp chinese-llama-lora-7b/*token* ${lora_model}/```

4. 将Chinese-LLaMA-LoRA中的adapter_config.json复制到${lora_model}

    ```cp chinese-llama-lora-7b/adapter_config.json ${lora_model}/```

5. 最后，编辑${lora_model}/adapter_config.json文件，修改其中的参数，确认其中的lora_alpha, r, modules_to_save, target_modules等参数与实际训练用的参数一致。

完成！现在${lora_model}可以用于后续的合并流程了。



# 用本地机器Fine Tune Alpaca-Plus 13B
你可以使用以下的命令在8x A100 (80GB)的机器上来训练Alpaca-Plus 13B。更新模型的权重路径`--model_name_or_path` 和 实际的数据路径`--data_path`。

```
torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ~/model_weights/Alpaca-Plus-13b  \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
