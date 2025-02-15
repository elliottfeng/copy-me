import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI

# 禁用 Streamlit 的文件监视器
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

def copy_me(api_key, query, knowledge_base_path="knowledge_base.txt"):
    """
    后端逻辑封装函数。

    参数:
        api_key (str): DeepSeek API 密钥。
        query (str): 用户输入的问题。
        knowledge_base_path (str): 知识库文件路径，默认为当前目录下的 knowledge_base.txt。

    返回:
        str: 生成的回答。
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"知识库文件 {knowledge_base_path} 不存在，请检查路径。")

        # 加载本地知识库
        loader = TextLoader(knowledge_base_path)
        documents = loader.load()

        # 文本分割
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # 向量化并存储
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 初始化 DeepSeek 大模型
        model = OpenAI(
            api_key=api_key,
            model_name="deepseek-chat",
            base_url="https://vip.apiyi.com/v1",
            max_tokens=1024,
        )

        # 实现 RAG 检索
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )
        result = qa.run(query)

        return result

    except FileNotFoundError as e:
        return f"错误：{e}"
    except Exception as e:
        return f"请求失败：{e}"