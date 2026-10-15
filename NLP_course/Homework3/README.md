# 中文到 Emoji：第一次训练

当前项目使用固定的150,000条中文—Emoji数据，从随机初始化训练手写、单层、单向、无Attention的LSTM Encoder–Decoder。教师侧数据生成与本地模型训练独立；训练和推理不调用大模型API。

## 目录导航

```text
emoji/
├── README.md             使用说明与目录导航
├── REPORT.md             第一次训练报告
├── train.py              从零训练入口
├── evaluate.py           最终测试入口
├── generate.py           中文转Emoji推理入口
├── run_experiment.py     完整实验流水线
├── src/                  模型、编码、指标及Emoji规范
├── scripts/              数据生成、验证、教师评测与打包工具
├── tests/                正确性与数据完整性检查
├── data/v2/              当前唯一数据、固定划分和标注约定
├── runs/first_training/  两种模型、日志、曲线与测试结果
├── docs/                 原始作业要求和Word实验报告
└── exports/              提交包、学生包、教师包及文件指纹
```

在项目根目录运行下面的命令。重新生成全部交付包使用 `.venv/bin/python scripts/finish_current.py`。教师包与完整提交包内容相同，均含测试参考与模型；只向学生分发学生包。`.env`仅保存在本地，不进入交付包。

## 环境与验证

Python 3.12，依赖见requirements-lock.txt。Apple GPU可用时使用MPS，否则使用CPU。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_dataset.py
```

Word版实验报告见 [中文到Emoji实验报告_第一次训练.docx](docs/中文到Emoji实验报告_第一次训练.docx)。

## 当前唯一数据集

数据位于data/v2。train/val/test为120,000/15,000/15,000，固定种子42，生成批次与已检出近重复句族整组隔离。Unicode Emoji完整单元统一编码，保留重复与顺序。标注约定见data/v2/ANNOTATION_GUIDE.md，剩余待复核候选见pending_review.jsonl；这不是全量人工验收的金标准。

学生仅获取exports/student_data_v2.zip，包含训练/验证标签和测试输入；exports/teacher_data_v2.zip含完整测试参考与模型，供教师保管。

## 第一次训练与复现

两种隐藏维度128/256从零训练，Embedding 128，Dropout 0.2，batch 256，Adam学习率0.001，梯度裁剪1.0，种子42，最多50轮，验证损失连续5轮不改善早停。中文及Emoji词表只从训练集建立。按最低验证交叉熵选检查点与最终模型，选定后才读测试参考。每轮完整验证集另做自主生成评测。

```bash
.venv/bin/python run_experiment.py --device auto
```

默认输出runs/first_training，已有配置时拒绝覆盖。再次复现请指定新的目录：

```bash
.venv/bin/python run_experiment.py --device auto --output runs/reproduction
```

单独执行等价命令：

```bash
.venv/bin/python train.py --hidden 128 --batch-size 256 --device auto --output runs/first_training/h128
.venv/bin/python train.py --hidden 256 --batch-size 256 --device auto --output runs/first_training/h256
.venv/bin/python evaluate.py --runs runs/first_training/h128 runs/first_training/h256 --device auto --output runs/first_training/final
.venv/bin/python scripts/make_report.py
.venv/bin/python scripts/package_submission.py
```

训练默认读取data/v2/prepared/train.jsonl及val.jsonl，最终测试默认读取同目录test.jsonl。数据保留固定指纹，不需要重新生成或重新划分。

## 推理与评分

读取runs/first_training/final/selection.json中的chosen目录作为推理模型：

```bash
.venv/bin/python generate.py --run runs/first_training/h256 --text '想喝咖啡' --device auto
```

例子中的h256应替换为实际chosen。训练使用完整Teacher Forcing，推理从BOS开始贪心解码并使用自己的上一步输出，遇EOS或长度上限停止。屏蔽PAD/UNK/BOS。

完全匹配要求顺序和重复次数一致。Emoji级micro Precision/Recall/F1先按样本内多重集合取交集再汇总，F1忽略顺序。未知参考符号保留参与评测，不映射为UNK伪造匹配。教师按学生提交ID对齐评分：

```bash
.venv/bin/python scripts/evaluate_v2_predictions.py --predictions student_predictions.jsonl --output runs/student_metrics.json
```

预测JSONL每行含id和解码后的emojis列表，测试ID完整且唯一，允许空预测，非法符号明确报错。

## 实现和产物

src/model.py手写LSTM门控、状态更新、时间步循环与自回归解码，未调用内置循环单元。最终h/c初始化解码器，Padding后同时冻结h/c。PyTorch仅提供基础张量、参数、自动求导及Adam。

runs/first_training下保存两种模型的权重、词表、配置、逐轮历史、验证输出与曲线；final保存最终选择、测试指标、全部测试案例、固定随机20条及最小差异诊断。REPORT.md由这些真实产物生成，exports/MANIFEST.json与exports/submission.zip用于交付及指纹核对。

旧数据、旧模型及其压缩包已清理；当前数据和本次第一次训练是项目唯一正式实验。数据文件保留v2目录名称作为固定来源标识，训练编号从第一次开始。
