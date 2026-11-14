# 作业五：BERT微调与中文情感分类

本实现依据`作业五-BERT微调与中文情感分类.docx`修改版，完成全参数微调、冻结编码器对照、测试指标、案例分析和PCA可视化。使用预训练中文BERT，不使用已针对情感分类微调的权重。

## 运行

在NLP_course目录执行，Python环境依赖见`Homework5/requirements.txt`。本次使用PyTorch 2.9.1 CUDA 12.8和BF16，需支持BF16的CUDA显卡。其他显卡如改用FP16需配合GradScaler。

```bash
pip install -r Homework5/requirements.txt
HF_ENDPOINT=https://huggingface.co CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python Homework5/code/run.py prepare
HF_ENDPOINT=https://huggingface.co CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python Homework5/code/run.py full
HF_ENDPOINT=https://huggingface.co CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python Homework5/code/run.py frozen
HF_ENDPOINT=https://huggingface.co CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python Homework5/code/run.py evaluate
CUDA_VISIBLE_DEVICES=0 python Homework5/code/predict.py "酒店很干净，服务很好。" "这本书并不好。"
```

`prepare`下载固定版本模型和数据，生成长度统计与编码示例。下载完成后可为后续命令添加`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`以避免重复联网；两个训练命令分别运行3轮；`evaluate`加载各自验证集Macro-F1最优模型，最终才使用测试集。代码内固定随机种子42、batch大小8、序列长度128、AdamW学习率2e-5、梯度裁剪1.0，不采用梯度累积。

## 产物

- `code/predict.py`：加载保存模型，对任意评论预测；CPU也可运行。
- `code/run.py`：数据处理、训练、评测、预测及绘图。
- `results/data_statistics.json`：样本数、类别分布、长度分位数与截断比例。
- `results/tokenization_example.json`：包含Padding的编码示例。
- `results/full/`和`results/frozen/`：配置、逐轮训练记录、资源统计、测试结果与图表。
- `test_predictions.jsonl`：每个测试样本原始行号id、text、label、prediction及两类概率。
- `random20.json`、`errors.json`、`custom_predictions.json`：随机20条、全部误判及自编评论预测。
- `reload_check.json`：固定输入重新加载后的预测一致性。
- `temp/results/full/checkpoint/`和`temp/results/frozen/checkpoint/`：模型及Tokenizer，仅保留本地，不提交Git。
- `temp/results/`：中间向量、长度数组、逐步训练日志、显存采样和控制台日志，不提交Git。
- `temp/`中的文档生成脚本仅作本地辅助工具，不随仓库发布；Word文档直接保存在Homework5目录。

新克隆的仓库不包含模型权重。先运行训练命令生成本地checkpoint，再执行评测或预测。重新运行时，代码会自动把中间结果写入temp；如需保存控制台日志，也应重定向到temp/results目录。

## 测量口径

训练时间为各轮训练循环耗时之和，包含batch准备、传输和日志操作；验证、保存权重单独计时。模型和数据下载、预处理、加载不计入训练时间。GPU与其他任务共享，耗时不能直接换算为独占GPU性能。

`resources.json`记录训练及验证阶段PyTorch峰值allocated、reserved显存，以及每秒采样一次的本进程nvidia-smi峰值。后者含CUDA上下文等额外占用，但可能错过短暂峰值；不把其他进程显存计入本实验。该记录是当前配置的实测峰值，不是所有batch大小或序列长度下的最大需求，也未通过故意触发OOM测设备极限。

冻结对照关闭编码器Dropout，仅更新分类头；其余训练参数与全参数微调一致，用于观察允许编码器更新的影响，不代表冻结方案调参后的最佳性能。PCA使用测试集pooler输出作事后可视化，不参与训练或模型选择。

## 来源

- 模型：https://huggingface.co/google-bert/bert-base-chinese
- 数据：https://huggingface.co/datasets/lansinuote/ChnSentiCorp
- 模型和数据版本哈希记录在代码与config.json中。
