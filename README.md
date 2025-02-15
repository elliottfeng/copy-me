# copy-me
Chat-Bot to introduce myself

疑难杂症：

1.弃用openai embeddings，因为API不一致

2.选用text2vec-base-chinese作为embeddings,原生中文支持

3.HuggingFace受网络阻碍，进行本地化尝试

4.尝试用对话历史来作为索引进行微调，但是新会话不保存记忆
# 保存历史对话到本地文件
def save_history(history, file_path="history.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# 从本地文件加载历史对话
def load_history(file_path="history.json"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 初始化历史记录
if "history" not in st.session_state:
    st.session_state.history = load_history()

5.尝试本地保存对话历史，但是加载新会话会把作为索引的对话重新加载到页面上

6.使用本地文档作为RAG
使用 LangChain 的 VectorStore 和 RetrievalQA 工具来实现这一点

