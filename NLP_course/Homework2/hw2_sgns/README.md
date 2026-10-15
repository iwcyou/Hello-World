# 作业二：SGNS 词向量与检索

完整实现更新后的基础要求及进阶任务；结果来自实际运行。核心训练只使用 PyTorch 张量、Embedding、自动求导和 Adam，没有调用 Word2Vec 库。

## 运行

Python 3.10+；本次实测 Python 3.13.9、PyTorch 2.9.1+cu128，详细版本见 `outputs/stats.json` 和完成报告。

```bash
cd Homework2/hw2_sgns
pip install -r requirements.txt
python train.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python analyze.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python verify.py
python ../temp/make_report.py
```

默认语料路径按脚本位置解析为 `Homework1/datasets/gene_related_papers_1000/`；自动使用 CUDA（如可用）。路径、参数可覆盖：

```bash
python train.py --corpus /path/to/gene_related_papers_1000 --device cpu --epochs 12 --output outputs_cpu
python analyze.py --output outputs_cpu
```

训练会覆盖指定输出目录中的同名产物；报告和 `verify.py` 默认读取 `outputs/`。词向量加载不需训练：

```python
from sgns import load_vectors, normalize
words, matrix = load_vectors('outputs')
unit = normalize(matrix)
i = words.index('cancer')
scores = unit @ unit[i]
scores[i] = -float('inf')
print([(words[j], float(scores[j])) for j in scores.argsort()[-5:][::-1]])
```

## 方法约定

- 所有 1000 篇论文的标题、摘要用于无监督训练；不使用正文，不划分泛化测试集。固定训练探针仅用于观察优化，不作为验证损失。
- 转小写，保留字母数字混合词（TP53→tp53）和内部连字符，移除标点、纯数字；保留希腊字母。源标题中序列化 XML 字典仅提取 `#text`。不做词干化；不同连字符、希腊字母与英文拼写未统一。
- 低于 5 次的词不建向量，原位置保留为 -1，构造窗口时跳过，不让过滤改变词间距离；不使用特殊词，不跨文档。窗口可跨句子。训练时不移除停用词、不做高频词下采样。
- 固定窗口半径 5，正样本每轮随机打乱。负词按词频的 0.75 次幂有放回采样；遇到当前正上下文词就重抽。允许重复负词、允许中心词本身为负词，其他邻域词也可能成为负词。这是噪声对比训练，不是语义上的真负例标注。重抽后实际分布是排除正上下文词的条件分布。
- 两张 5373×100 矩阵；sigmoid 为二分类激活，`logsigmoid` 稳定计算损失。5 个负项求和，再按正样本取平均。导出中心词矩阵。
- 300 个高频候选词用于聚类，过滤 scikit-learn 英文停用词及 `analyze.py` 中显式列出的通用词；在归一化原始向量上 K-means，K=8、n_init=20、seed=42。K=8 是便于解释和展示的预设，并非最优簇数。
- 每簇取频率最高的 10 个词（本次共 80 个）做 t-SNE，perplexity=20、PCA 初始化、1500 次迭代、seed=42；聚类先于降维。每簇代表词按与归一化簇中心的余弦相似度排序。
- BM25 自行实现（k1=1.5、b=0.75）；词向量检索使用 TF×平滑 IDF 加权平均并归一化，IDF=log((N+1)/(df+1))+1，两端同样编码。OOV 忽略，全 OOV 查询返回空结果；零向量文档与查询得分为 0。
- 两种检索使用相同 1000 篇标题+摘要、相同分词和固定查询，均无监督、均不调查询参数。BM25 可保留低频词；SGNS 受最低词频限制，这一信息差异是基线局限。预先编写的 `queries.json` 含 6 篇指定论文各两条查询（词面式／改写式），仅将指定原文当作目标，没有穷尽其他相关论文。因此 target Recall@5 等于 Hit@5，MRR@5 只按该目标计算；不能解释为开放主题检索上的完整相关性评测。

## 文件

| 文件 | 作用 |
|---|---|
| `sgns.py` | 分词、词表、窗口、负采样、SGNS、加载 |
| `train.py` | 语料读取、训练、配置与权重保存 |
| `analyze.py` | 相似词、t-SNE、聚类、BM25 和向量检索 |
| `queries.json` | 预先固定的查询与目标文档 |
| `tests/test_sgns.py` / `verify.py` | 7 项正确性测试、实际产物重载检查 |
| `../temp/make_report.py` | 根据实测产物生成 Word 报告 |
| `outputs/vocab.json` | 按 ID 排列的词、词频及 word_to_id |
| `outputs/model.pt` / `vectors.npy` | 两张矩阵的模型权重／中心词矩阵 |
| `outputs/config.json` / `stats.json` / `history.json` | 参数、统计、环境、耗时和损失 |
| `outputs/manifest.json` / `documents.json` | 每份输入文件 SHA256／使用的标题摘要 |
| `outputs/neighbors.json` / `clusters.json` / `tsne_points.json` | 相似词及分数、完整聚类、二维坐标 |
| `outputs/retrieval.json` | 各查询 Top-5、OOV 及分组指标 |
| `outputs/loss.png` / `tsne_clusters.png` | 报告插图 |
| `outputs/sample_pairs.json` / `tests.txt` / `verification.json` | 示例样本及检查证据 |

固定种子和确定性算法用于复现；跨硬件、PyTorch / scikit-learn 版本可能存在浮点与布局差异。没有设置训练损失或相似词结果的“必达”阈值。训练和查询集均来自本语料，不主张跨语料泛化能力。
