# 在线查询 API

在线模块负责“查询向量化 → Qdrant 召回 → Prompt 编排 → LLM 流式生成”。
HTTP 接口使用 FastAPI，查询结果通过 SSE（Server-Sent Events）流式返回。

## 启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
set -a
source .env
set +a

uvicorn api.app:app --host 0.0.0.0 --port 8080
```

启动后：

- API：`http://localhost:8080`
- OpenAPI：`http://localhost:8080/docs`
- 健康检查：`GET /health`
- 流式问答：`POST /v1/query`

至少需要配置：

```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise_knowledge_bge_m3
EMBEDDING_MODEL=BAAI/bge-m3

LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=
```

Embedding 和 LLM 都支持 OpenAI 兼容服务。如果 Embedding 使用企业网关，
设置 `EMBEDDING_PROVIDER=openai_compatible`、`EMBEDDING_API_BASE_URL` 和
`EMBEDDING_DIMENSION`。

## 流式查询

```bash
curl -N -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "报销需要准备哪些材料？",
    "top_k": 5,
    "filters": {"file_type": ["pdf", "markdown"], "chunk_role": "child"},
    "include_sources": true
  }'
```

响应类型为 `text/event-stream`，事件顺序如下：

| 事件 | 说明 |
| --- | --- |
| `start` | 查询开始，包含问题和实际 `top_k` |
| `sources` | 召回的知识块，包含来源、页码、分数和正文 |
| `delta` | LLM 生成的增量文本，可重复出现 |
| `done` | 生成结束，包含 `finish_reason` 和耗时 |
| `error` | 查询失败，包含稳定的错误码和用户可读消息 |

示例：

```text
event: start
data: {"query":"报销需要准备哪些材料？","top_k":5}

event: sources
data: {"items":[{"index":1,"score":0.82,"source":"/knowledge/policy.pdf","page_number":3,"content":"..."}]}

event: delta
data: {"content":"根据"}

event: delta
data: {"content":"资料，报销需要"}

event: done
data: {"finish_reason":"stop","elapsed_ms":1832.4}
```

如果没有召回任何知识块，服务不会调用 LLM，而是返回固定兜底文本，并将
`finish_reason` 设为 `no_context`。

## 请求参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question` | string | 必填，最大 8000 字符 |
| `top_k` | integer | 召回数量，默认取 `RAG_TOP_K` |
| `score_threshold` | number | 最低相似度分数，默认取 `RAG_SCORE_THRESHOLD` |
| `filters` | object | Qdrant payload 过滤条件，值可为标量或数组 |
| `include_sources` | boolean | 是否发送 `sources` 事件，默认 true |
| `retrieval_mode` | string | `hybrid` 或 `dense`，默认取 `RAG_RETRIEVAL_MODE` |
| `temperature` | number | 覆盖 LLM 温度，范围 0 到 2 |
| `max_tokens` | integer | 覆盖最大生成 token 数 |

常用过滤字段包括 `document_id`、`source`、`file_type`、`chunk_strategy`、
`chunk_role` 和 `embedding_model`。父子分块场景通常过滤
`chunk_role=child`。

默认使用 `hybrid`：dense 向量召回和 Qdrant BM25 文本召回各取候选，再通过
RRF 融合。融合结果的 `score` 是 RRF 排名分，不是余弦相似度；`sources`
事件会同时返回 `retrieval_mode`、`route_scores` 以及每路原始分数。文本索引
不可用或旧 collection 尚未重建时，会自动回退到 dense 检索。

## Python 客户端

```python
import json

import httpx


with httpx.stream(
    "POST",
    "http://localhost:8080/v1/query",
    json={"question": "差旅报销标准是什么？", "top_k": 5},
    timeout=120,
) as response:
    response.raise_for_status()
    event_name = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data = json.loads(line.removeprefix("data: "))
            if event_name == "delta":
                print(data["content"], end="", flush=True)
            elif event_name == "sources":
                print("\nsources:", data["items"])
```

## 代码结构

```text
api/
├── app.py                    # FastAPI 应用与运行时装配
├── QueryRequest.py           # 请求模型
├── sse.py                    # SSE 编码
└── routes/
    ├── health.py             # 健康检查
    └── query.py              # 流式查询路由
generation/
├── BaseLLM.py                # 流式 LLM 接口
├── OpenAICompatibleLLM.py    # OpenAI 兼容 Chat Completions
└── LLMError.py
retrieval/
├── KnowledgeRetriever.py     # 查询向量化与召回
├── PromptBuilder.py          # 引用编号上下文与系统 Prompt
├── RagQueryService.py        # 在线 RAG 流式编排
└── RagStreamEvent.py         # 流式事件
```

## 生产注意事项

- API 本身未内置鉴权，建议放在企业网关或反向代理后，并启用 TLS。
- 使用本地 bge-m3 和 GPU 时，每个 uvicorn worker 都会加载一份模型；按显存
  控制 worker 数量。Embedding 也可切换到独立的企业 OpenAI 兼容服务。
- SentenceTransformer 首次查询会加载模型；生产环境建议预热模型缓存。
- `API_CORS_ORIGINS` 为空时不启用 CORS。多个来源用逗号分隔，禁止长期使用
  `*` 开放生产接口。
- 当前只提供单轮问答。多轮会话可后续在 `PromptBuilder` 和请求模型中增加
  `messages/history`，无需改动检索与 SSE 协议。
