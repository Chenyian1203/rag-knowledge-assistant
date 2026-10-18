# 🧠 企业级智能知识库助手 (RAG)

基于 Streamlit、LangChain 和 Chroma 构建的私有知识库问答助手，支持多格式文档解析与多轮对话记忆。

## ✨ 核心特性
- **多格式支持**：无缝解析 PDF、TXT、Word 文档。
- **多轮对话记忆**：能记住上下文，像真人一样交流。
- **精准检索**：基于向量数据库，从文档中寻找最相关的答案，杜绝大模型“幻觉”。
- **工程化规范**：采用 `.env` 隔离敏感密钥，保护 API 安全。

## 🛠️ 技术栈
- 前端：Streamlit
- 核心框架：LangChain
- 向量数据库：ChromaDB
- 大模型：DeepSeek API

## 🚀 快速启动
1. 安装依赖：`pip install streamlit langchain langchain-openai langchain-community chromadb pypdf`
2. 配置密钥：在根目录新建 `.env` 文件，填入 `OPENAI_API_KEY=你的Key`
3. 运行项目：`streamlit run app.py`
