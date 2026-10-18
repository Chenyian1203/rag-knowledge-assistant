__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# 加载本地 .env 文件里的密钥（更安全）
load_dotenv()

st.set_page_config(page_title="智能知识库助手", page_icon="🧠", layout="wide")
st.title("🧠 企业级智能知识库助手")
st.caption("支持 PDF / TXT / Word，具备多轮对话记忆能力")

# 初始化聊天记录
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("你好！请上传文档，然后向安安提问鸭。")

with st.sidebar:
    st.header("⚙️ 设置")
    uploaded_file = st.file_uploader("上传文档", type=["pdf", "txt", "docx"])
    if st.button("🗑️ 清空对话"):
        msgs.clear()
        st.rerun()

# 处理文档
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    file_path = f"temp.{file_extension}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    @st.cache_resource(show_spinner="正在解析文档并构建知识库哦要耐心等待喵...")
    def process_doc(path, ext):
        if ext == 'pdf':
            loader = PyPDFLoader(path)
        elif ext == 'docx':
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path, encoding='utf-8')
            
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        embeddings = FakeEmbeddings(size=1536)
        # 清空旧库，重新建立
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = process_doc(file_path, file_extension)

    # 构建带记忆的 RAG 链
    llm = ChatOpenAI(model="deepseek-chat", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个严谨的助手。请严格根据以下背景信息回答。如果不知道，就说不知道。\n\n背景信息：{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 将检索到的文档和问题组合
    def get_context(question):
        return format_docs(retriever.invoke(question))

    chain = (
        {"context": lambda x: get_context(x["question"]), "question": lambda x: x["question"], "history": lambda x: x["history"]}
        | prompt | llm | StrOutputParser()
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 渲染聊天界面
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_query := st.chat_input("请输入你的问题..."):
        st.chat_message("human").write(user_query)
        with st.chat_message("ai"):
            with st.spinner("思考中..."):
                response = conversational_rag_chain.invoke(
                    {"question": user_query},
                    config={"configurable": {"session_id": "any"}}
                )
                st.write(response)

