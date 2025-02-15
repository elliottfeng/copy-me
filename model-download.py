import streamlit as st
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# 定义保存历史记录的 JSON 文件路径
HISTORY_FILE = "chat_history.json"

# 初始化历史记录
if "history" not in st.session_state:
    # 如果本地有历史记录文件，则加载
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

# 初始化当前会话记录
if "current_session" not in st.session_state:
    st.session_state.current_session = []

# 初始化 ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",  # 替换为你的模型名称
    api_key="sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43",  # 替换为你的 API 密钥
    base_url="https://vip.apiyi.com/v1",  # 替换为你的 API 地址
    max_tokens=1024
)

# 使用本地模型初始化 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="./text2vec-base-chinese")


# 初始化向量存储
def init_vector_store(history):
    # 将历史对话信息转换为文本
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # 使用文本分割器将历史对话信息分割成小块
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(history_text)

    # 创建向量存储
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


# 初始化 RetrievalQA
def init_retrieval_qa(llm, vector_store):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa


# 获取模型回复和检索内容
def get_response_material(query, history):
    # 初始化向量存储
    vector_store = init_vector_store(history)

    # 初始化 RetrievalQA
    qa = init_retrieval_qa(llm, vector_store)

    # 调用 RetrievalQA 生成回复
    response = qa.run(query)

    # 检索相关内容
    material = retrieve_material(query)

    return response, material



# 保存历史记录到本地文件
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


# 显示当前会话记录
def show_current_session():
    for message in st.session_state.current_session:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# 接收用户输入
if user_input := st.chat_input("Chat with 冯宇洋: "):
    # 在页面上显示用户的输入
    with st.chat_message("user"):
        st.markdown(user_input)

    # 将用户的输入加入当前会话记录
    st.session_state.current_session.append({"role": "user", "content": user_input})

    # 获取模型生成的回复和检索内容
    response, material = get_response_material(user_input, st.session_state.history)

    # 使用一个左侧框，展示检索到的信息
    with st.sidebar:
        st.markdown(
            ':balloon::tulip::cherry_blossom::rose: :green[**检索内容：**] :hibiscus::sunflower::blossom::balloon:')
        st.text(material)

    # 在页面上显示模型生成的回复
    with st.chat_message("assistant"):
        st.markdown(response)

    # 将模型的输出加入到当前会话记录
    st.session_state.current_session.append({"role": "assistant", "content": response})

    # 将当前会话记录加入历史记录
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})

    # 只保留最近的 20 轮对话
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

    # 保存历史记录到本地文件
    save_history(st.session_state.history)

# 显示当前会话记录
show_current_session()