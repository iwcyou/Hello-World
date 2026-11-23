# 作业六：RAG中的检索与重排

本目录包含作业要求、学生版数据集和一份参考实现。学生版数据包只有`corpus.jsonl`、`query.jsonl`、`label.jsonl`三个文件；数据字段、评测指标和提交内容以Word作业要求为准。作业不要求答案生成。

参考实现使用[BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)编码查询和文档片段，使用[MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)重排候选。模型版本记录在`code/rag.py`。FAISS索引及运行结果保存在`temp/`和`results/`，不随作业发布。

## 运行

在`Homework6`目录执行。首次运行会下载模型。下列命令依次构建索引、召回、融合、重排，并评测四组消融方案：

```bash
pip install -r requirements.txt
mkdir -p temp/student
unzip -o 作业六-RAG数据集.zip -d temp/student
export HF_ENDPOINT=https://huggingface.co
python code/rag.py build
python code/rag.py retrieve
python code/hybrid.py
for stage in hybrid retrieve bm25; do
    python code/rag.py rerank --candidate-stage "$stage"
done
for run in hybrid_rerank retrieve_rerank bm25_rerank hybrid; do
    python code/evaluate.py temp/student/corpus.jsonl temp/student/label.jsonl "results/${run}.jsonl"
done
```

向量检索模型同时编码文档和查询。每条图注与段落独立编码，文档向量与查询向量均做L2归一化，再用FAISS内积索引检索。BM25与向量检索各取前50条，按RRF融合后保留50条；重排模型只调整候选顺序。需要降低显存占用时可为`rag.py`增加`--batch-size 16`，也可使用`--device cpu`。

自然问题检索实验须自行选取三条`recaption`并编写英文问题，使用同一套索引和模型检索；新问题由人工分析，不按原标注计算指标。
