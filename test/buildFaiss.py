from langchain_community.vectorstores import FAISS
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
import os

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# 递归加载文档，并将相对路径写入metadata
def load_documents_from_folder(folder_path: str):
    docs = []
    for root, _, files in os.walk(folder_path):  # 递归扫描
        for filename in files:
            file_path = os.path.join(root, filename)

            # 按扩展名选择loader
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                loader = UnstructuredExcelLoader(file_path)
            else:
                continue

            # 加载文件
            loaded_docs = loader.load()

            # 在 metadata 中记录相对路径（从根目录开始）
            rel_path = os.path.relpath(file_path, folder_path)
            for d in loaded_docs:
                d.metadata["source"] = rel_path

            docs.extend(loaded_docs)
    return docs

# 构建FAISS向量库
def build_faiss_index_from_folder(folder_path: str, index_save_path: str):
    print("📂 加载法规文档中...")
    raw_docs = load_documents_from_folder(folder_path)

    print(f"📄 共加载 {len(raw_docs)} 个原始文档片段")

    print("✂️ 切分文档为片段...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(raw_docs)

    print(f"📄 切分后得到 {len(split_docs)} 个片段")

    print("🔍 构建嵌入向量...")
    embeddings = embedding_model

    print("💾 构建 FAISS 向量数据库...")
    vectordb = FAISS.from_documents(split_docs, embeddings)

    print(f"✅ 保存向量数据库至：{index_save_path}")
    vectordb.save_local(index_save_path)
    return vectordb


if __name__ == "__main__":
    folder = "./test/rule_files"
    save_path = "./test/faiss_law_index"
    build_faiss_index_from_folder(folder, save_path)
