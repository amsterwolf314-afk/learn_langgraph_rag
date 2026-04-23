# Custom RAG with LangGraph

一个基于 LangGraph 的完整 Python RAG 示例项目，支持直接使用 `langgraph dev` 启动本地 Agent Server。项目会抓取 Lilian Weng 的公开博客文章，构建本地向量缓存，并通过“问题路由 -> 检索 -> 相关性判断 -> 问题改写 -> 生成答案”的图工作流完成问答。

## What This Project Includes

- 完整的 `langgraph.json` 配置
- 可直接安装的 `requirements.txt`
- 可被 LangGraph CLI 加载的图入口 `./src/rag.py:graph`
- 本地 `.env` 环境变量加载
- 可复用的向量缓存构建逻辑
- 兼容 `langgraph dev` 的项目目录结构

## Project Structure

```text
.
├── .env.example
├── langgraph.json
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_answer.py
│   ├── message_utils.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── rewrite_question.py
│   └── workflow.py
└── README.md
```

## Requirements

- Python 3.11
- 可用的 LLM API
- 可用的 Embedding API
- 可选但推荐的 LangSmith API Key，用于本地 LangGraph 开发体验与可观测性

## Installation

创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -U pip
pip install -r requirements.txt
```

## Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

然后填写你的真实配置：

```env
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.minimaxi.com/v1
LLM_MODEL_ID=MiniMax-M2.7
LLM_TIMEOUT=60
USER_AGENT=custom-rag-langgraph/1.0
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
EMBED_API_KEY=your_embed_api_key_here
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

当前项目启动时至少依赖这些变量：

- `LLM_API_KEY`
- `LLM_MODEL_ID`
- `EMBED_API_KEY`

## Run With LangGraph Dev

先验证项目配置：

```bash
langgraph validate
```

启动本地开发服务：

```bash
langgraph dev
```

默认启动后可访问：

- API: `http://localhost:2024`
- Docs: `http://localhost:2024/docs`

## Quick Test

服务启动后，可以直接请求本地运行的 graph：

```bash
curl -s --request POST \
  --url "http://localhost:2024/runs/stream" \
  --header "Content-Type: application/json" \
  --data '{
    "assistant_id": "agent",
    "input": {
      "messages": [
        {
          "role": "human",
          "content": "What does Lilian Weng say about reward hacking?"
        }
      ]
    },
    "stream_mode": "messages-tuple"
  }'
```

也可以继续使用模块方式做本地调试：

```bash
python -m src.retrieval
python -m src.rag
```

## How It Works

1. `generate_query_or_respond` 决定直接回答还是调用检索工具。
2. `retrieve_blog_posts` 从本地缓存向量库检索相关内容。
3. `grade_documents` 判断检索结果是否足够相关。
4. `rewrite_question` 在检索不佳时重写问题并重试。
5. `generate_answer` 基于检索上下文生成最终回答。

## Notes

- 向量缓存会写入 `.cache/`，默认不会提交到 Git。
- `langgraph.json` 使用的是 requirements 路线配置，适合当前这个轻量 Python 项目。
- 代码中的导入方式已调整为更适合 LangGraph CLI 加载图文件的绝对导入。

## Privacy And Security

- `.env`、`.cache/`、`.langgraph_api/` 已在 `.gitignore` 中忽略
- 不要把真实密钥提交到仓库、截图、issue 或 README
- 如果真实 API Key 曾暴露在聊天记录、终端回显或错误提交中，请尽快轮换
