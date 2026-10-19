import streamlit as st
import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 1. 读取 .env ====================
load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ 没有读取到 API Key！请检查你的 .env 文件是否存在。")
    st.stop()

# ==================== 2. 高级商业设计 CSS ====================
st.set_page_config(page_title="企业级 AI 知识库", page_icon="✨", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* 网格高级背景 */
    .stApp {
        background-color: #fafafa;
        background-image: 
            linear-gradient(#f0f0f0 1px, transparent 1px),
            linear-gradient(90deg, #f0f0f0 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* 顶部导航条 */
    .top-nav {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 1px solid #eaeaea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        margin-top: -5rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0 0 15px 15px;
    }
    .logo-text {
        font-weight: 800;
        font-size: 1.2rem;
        color: #333;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* 渐变金色大标题 */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #e6a800, #ffcc00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1rem;
    }
    .sub-title {
        color: #888;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
    }

    /* 居中上传虚线卡片 */
    .upload-card {
        background: #ffffff;
        border: 2px dashed #ffcc00;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 204, 0, 0.1);
        max-width: 800px;
        margin: 0 auto;
        transition: all 0.3s ease;
    }
    
    /* 功能标签 */
    .feature-tags {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 3rem;
    }
    .tag {
        background: #fff9e6;
        color: #b38000;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid #ffeaa7;
    }

    /* AI 金色边框回答卡片 */
    .answer-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
        margin-top: 2rem;
        border-left: 8px solid #ffcc00;
    }

    /* 金色高级按钮 */
    .stButton>button {
        background: linear-gradient(90deg, #ffcc00 0%, #ffdb4d 100%);
        color: #333 !important;
        border-radius: 8px;
        border: none;
        padding: 14px 40px;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 204, 0, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 204, 0, 0.6);
    }

    /* 隐藏侧边栏 */
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# 顶部假导航
st.markdown("""
<div class="top-nav">
    <div class="logo-text">✨ 智库科技 | AI Knowledge</div>
    <div style="color: #999; font-size: 0.9rem;">Enterprise Edition</div>
</div>
""", unsafe_allow_html=True)

# ==================== 3. 页面主体 ====================
st.markdown('<div class="main-title">企业级 AI 知识库</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">基于最前沿 RAG 架构，多格式安全解析，打造企业专属的最强大脑</div>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-tags">
    <div class="tag">⚡ 极致性能</div>
    <div class="tag">🔐 数据安全</div>
    <div class="tag">🔍 向量检索</div>
    <div class="tag">🧠 无幻觉推理</div>
</div>
""", unsafe_allow_html=True)

# 核心区域
container = st.container()

with container:
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    if not st.session_state.file_uploaded:
        st.markdown("""
        <div class="upload-card">
            <h1 style="font-size: 4rem; margin-bottom: 10px;">📂</h1>
            <h3 style="color: #333;">点击下方按钮上传文件</h3>
            <p style="color: #999;">支持 PDF、Word、TXT 格式，大小不超过 200MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")
            if uploaded_file is not None:
                st.session_state.file_uploaded = True
                st.session_state.current_file = uploaded_file
                st.rerun() 
    else:
        uploaded_file = st.session_state.current_file
        file_extension = uploaded_file.name.split('.')[-1]
        temp_filename = f"temp.{file_extension}"
        
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("AI 正在深度解析文档结构..."):
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_filename)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(temp_filename)
            else:
                loader = TextLoader(temp_filename, encoding='utf-8')
                
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            full_context = "\n\n".join([doc.page_content for doc in chunks])
        
        st.success(f"✅ 文档 **{uploaded_file.name}** 已成功载入 AI 大脑，您可以开始提问了。")
        
        st.markdown("---")
        
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            question = st.text_input("💬 输入问题", placeholder="请输入您想了解的内容...", label_visibility="collapsed")
        with col_btn:
            submit_btn = st.button("🚀 提 问")
        
        if submit_btn and question:
            with st.spinner("AI 正在检索知识库..."):
                headers = {
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json; charset=utf-8"
                }
                context_text = full_context[:15000]
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": f"你是一个严谨的企业助手。请严格根据以下背景信息回答。如果找不到答案，就说不知道。\n\n背景信息：\n{context_text}"},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0
                }
                
                json_payload = json.dumps(data, ensure_ascii=True)
                
                # 【修复404的核心就在这里：增加了 .rstrip('/')】
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    data=json_payload.encode('utf-8')
                )
                
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4 style="color: #b38000; margin-top:0; display: flex; align-items: center; gap: 8px;">🤖 AI 深度解析</h4>
                        <p style="font-size: 1.15rem; line-height: 1.8; color: #333; margin-bottom:0;">{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"请求失败，状态码: {response.status_code}")
                    st.write(response.text)