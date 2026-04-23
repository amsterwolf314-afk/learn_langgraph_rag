# Custom RAG with LangGraph

一个基于 LangGraph 的轻量级 RAG 示例项目。它会从 Lilian Weng 的几篇博客中抓取内容，构建本地向量缓存，并通过检索、重写问题、生成答案这几个节点组成一个简单工作流。

## Features

- 基于 LangGraph 编排问答流程
- 使用 OpenAI-compatible Chat Model 进行问题路由、改写与回答生成
- 使用 Embedding 模型构建本地向量缓存
- 自动根据检索结果判断是否需要重写问题
- 向量库缓存到本地，避免重复抓取和重复嵌入

## Project Structure

```text
.
├── langgraph.json
├── src/
│   ├── config.py
│   ├── retrieval.py
│   ├── workflow.py
│   ├── rewrite_question.py
│   ├── generate_answer.py
│   └── rag.py
└── .env.example
```

## Requirements

- Python 3.10+
- 可用的 OpenAI-compatible LLM API
- 可用的 Embedding API

建议先创建虚拟环境，再安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install langgraph langchain langchain-openai langchain-community langchain-text-splitters python-dotenv pydantic beautifulsoup4 tiktoken
```

## Environment Variables

先复制环境变量模板：

```bash
cp .env.example .env
```

然后填写这些配置：

```env
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.minimaxi.com/v1
LLM_MODEL_ID=MiniMax-M2.7
LLM_TIMEOUT=60
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
EMBED_API_KEY=your_embed_api_key_here
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

当前代码实际依赖的关键变量是：

- `LLM_API_KEY`
- `LLM_MODEL_ID`
- `EMBED_API_KEY`

## Run

构建或刷新本地向量缓存：

```bash
python -m src.retrieval
```

运行图工作流示例：

```bash
python -m src.rag
```

如果你使用 LangGraph 本地开发工具，也可以基于 [langgraph.json](./langgraph.json) 启动。

## Privacy And Security

- `.env`、`.cache/`、`.langgraph_api/` 已加入 `.gitignore`，不会默认提交到仓库
- 不要把真实 API Key 写入 `.env.example`、README、截图或 issue
- 如果密钥曾出现在聊天记录、终端共享、录屏或错误提交中，请立即到对应平台轮换
- 本项目会抓取公开网页并将文本缓存到本地 `.cache/`，如需公开分发仓库，建议不要提交这些缓存文件

## Notes

- 向量缓存依赖 `BLOG_URLS`、分块参数和 Embedding 配置；配置变化后会自动触发重建
- 当前示例数据源固定为 Lilian Weng 的博客文章，适合学习 LangGraph 中的基础 RAG 编排
