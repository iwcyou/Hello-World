import os
import torch
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.docstore.document import Document

# ===== 1. 配置嵌入模型 =====
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # 官方建议归一化
)

# ===== 2. 加载文档 =====
def load_documents_from_folder(folder_path: str):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"读取 PDF 失败: {filename}, 错误: {e}")

        elif filename.lower().endswith(".docx"):
            try:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"读取 Word 失败: {filename}, 错误: {e}")

        elif filename.lower().endswith((".xlsx", ".xls")):
            try:
                # 读取所有 sheet，转换为字符串
                df_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl", dtype=str)
                for sheet_name, sheet_df in df_sheets.items(): 
                    try:
                        sheet_df = sheet_df.fillna("")
                        text = sheet_df.to_string(index=False)
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": f"{filename} - {sheet_name}"}
                        ))
                    except Exception as e:
                        print(f"⚠️ Sheet 读取失败: {filename} - {sheet_name}, 错误: {e}")
            except Exception as e:
                print(f"读取 Excel 失败: {filename}, 错误: {e}")

        else:
            print(f"跳过不支持的文件类型: {filename}")

    return docs

# ===== 3. 构建 FAISS 索引 =====
def build_faiss_index_from_folder(folder_path: str, index_save_path: str):
    print("📂 正在加载法规文档...")
    raw_docs = load_documents_from_folder(folder_path)
    print(f"📄 共加载 {len(raw_docs)} 个文档片段")

    if not raw_docs:
        print("❌ 未加载到任何文档，请检查文件夹路径或文件格式")
        return None

    print("✂ 正在切分文档为片段...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"📄 切分后得到 {len(split_docs)} 个文档块")

    if not split_docs:
        print("❌ 没有文档块可用于构建索引，请检查文件是否为空")
        return None

    print("🔍 正在构建嵌入向量...")
    embeddings = embedding_model

    print("📦 正在构建 FAISS 向量数据库...")
    vectordb = FAISS.from_documents(split_docs, embeddings)

    print(f"💾 保存向量数据库至：{index_save_path}")
    os.makedirs(index_save_path, exist_ok=True)
    vectordb.save_local(index_save_path)

    print("✅ 向量数据库构建完成！")
    return vectordb


if __name__ == "__main__":
    # ======== 配置路径 ========
    docs_folder = "./test/rule"       # 你的法规文档所在文件夹
    index_folder = "./faiss_index"    # 保存 FAISS 索引的文件夹

    # 构建索引
    build_faiss_index_from_folder(docs_folder, index_folder)
