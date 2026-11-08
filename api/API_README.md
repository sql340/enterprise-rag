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

# 启用 ACL 时必填；由可信网关注入同一个共享密钥
RAG_ACL_ENABLED=true
RAG_ACL_PROXY_TOKEN=replace-with-a-strong-random-secret

# Graph RAG：默认参与 RRF 融合；未构建图谱时自动退化为 dense + BM25
RAG_RETRIEVAL_MODE=hybrid_graph
RAG_GRAPH_ENABLED=true
RAG_GRAPH_STORE_PATH=.cache/knowledge_graph.sqlite3
RAG_GRAPH_WEIGHT=0.8
RAG_GRAPH_MAX_HOPS=1
```

Embedding 和 LLM 都支持 OpenAI 兼容服务。如果 Embedding 使用企业网关，
设置 `EMBEDDING_PROVIDER=openai_compatible`、`EMBEDDING_API_BASE_URL` 和
`EMBEDDING_DIMENSION`。

## 流式查询

```bash
curl -N -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -H "X-Rag-Proxy-Token: $RAG_ACL_PROXY_TOKEN" \
  -H 'X-Tenant-Id: tenant-a' \
  -H 'X-User-Id: user-1' \
  -H 'X-User-Roles: finance,viewer' \
  -H 'X-User-Groups: audit' \
  -H 'X-User-Clearance: internal' \
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
| `retrieval_mode` | string | `dense`、`hybrid`、`graph` 或 `hybrid_graph`，默认取 `RAG_RETRIEVAL_MODE` |
| `temperature` | number | 覆盖 LLM 温度，范围 0 到 2 |
| `max_tokens` | integer | 覆盖最大生成 token 数 |

常用过滤字段包括 `document_id`、`source`、`file_type`、`chunk_strategy`、
`chunk_role` 和 `embedding_model`。父子分块场景通常过滤
`chunk_role=child`。

`filters` 不允许包含 `tenant_id`、`classification`、`allowed_roles`、
`allowed_groups`、`allowed_users`、`owner_id`、`document_status` 或
`acl_version`；这些字段只能由服务端认证上下文生成，否则请求返回 422。

默认使用 `hybrid_graph`：dense 向量召回、Qdrant BM25 文本召回和图谱实体召回
各取候选，再通过 RRF 融合。融合结果的 `score` 是 RRF 排名分，不是余弦相似度；
`sources` 事件会同时返回 `retrieval_mode`、`route_scores` 以及每路原始分数。
文本索引或图谱不可用时，会自动使用其余可用检索路；全部增强路不可用时退化为
dense 检索。

各模式含义：

| 模式 | 说明 |
| --- | --- |
| `dense` | 仅向量相似度召回 |
| `hybrid` | dense + BM25，通过 RRF 融合 |
| `graph` | 仅实体关系图谱召回 |
| `hybrid_graph` | dense + BM25 + graph 三路融合，默认模式 |

Graph RAG 需要先用 `GraphIndexer` 构建实体关系。图谱只返回候选 `chunk_id`，
服务端随后通过 Qdrant 的 ACL 过滤读取正文；知识图谱路由不会绕过租户和文档权限。
构建方式见 [`../docs/GRAPH_RAG.md`](../docs/GRAPH_RAG.md)。

## 身份与知识权限

`RAG_ACL_ENABLED=true` 时，查询接口要求以下可信身份头：

| 请求头 | 说明 |
| --- | --- |
| `X-Rag-Proxy-Token` | 网关与 RAG 服务共享的密钥 |
| `X-Tenant-Id` | 租户 ID，必须与知识块 `tenant_id` 完全一致 |
| `X-User-Id` | 用户 ID，大小写敏感 |
| `X-User-Roles` | 逗号分隔的角色，服务端统一转小写 |
| `X-User-Groups` | 逗号分隔的用户组，服务端统一转小写 |
| `X-User-Clearance` | `public`、`internal`、`confidential` 或 `secret` |
| `X-User-Name` | 可选展示名 |

身份头只有在可信网关已经完成认证、剥离客户端伪造值并重新注入时才可信任。
直接暴露 API 端口会破坏这一前提。`RAG_ACL_PROXY_TOKEN` 未配置时，ACL 查询会
返回 503，服务不会退回到“信任任意请求头”的模式。

仅在本地调试或迁移旧系统时可以设置 `RAG_ACL_ENABLED=false` 临时关闭 ACL；
生产环境不应关闭。

文档和知识块的权限关系为：

1. `tenant_id` 必须完全一致。
2. 用户 `clearance` 必须大于等于文档 `classification`。
3. 文档状态必须为 `active`。
4. 非管理员还必须满足 owner、用户白名单、角色白名单、用户组白名单或
   `public` 密级中的至少一项。

`knowledge_admin` 和 `platform_admin` 可以跳过第 4 条，但仍不能跨租户、越密级
读取或读取已归档/删除文档。知识块继承文档 ACL；ACL 变更后可通过
`KnowledgeIndexer.update_acl_by_document()` 原地更新 payload，不必重算向量。

ACL 启用前已经入库、没有 `tenant_id` 和 ACL payload 的旧知识块默认不可见。
需要为这些文档补齐 ACL 并重新入库，不能依赖兼容模式放宽过滤。

## Python 客户端

```python
import json
import os

import httpx


headers = {
    "X-Rag-Proxy-Token": os.environ["RAG_ACL_PROXY_TOKEN"],
    "X-Tenant-Id": "tenant-a",
    "X-User-Id": "user-1",
    "X-User-Roles": "finance,viewer",
    "X-User-Groups": "audit",
    "X-User-Clearance": "internal",
}
with httpx.stream(
    "POST",
    "http://localhost:8080/v1/query",
    headers=headers,
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
access_control/
├── Classification.py         # 文档密级
├── Principal.py              # 已认证主体
├── PrincipalResolver.py      # 可信身份头解析
├── DocumentAcl.py            # 文档 ACL 与访问策略
└── QdrantAclFilter.py        # Qdrant 强制访问过滤条件
generation/
├── BaseLLM.py                # 流式 LLM 接口
├── OpenAICompatibleLLM.py    # OpenAI 兼容 Chat Completions
└── LLMError.py
retrieval/
├── KnowledgeRetriever.py     # 查询向量化与召回
├── PromptBuilder.py          # 引用编号上下文与系统 Prompt
├── RagQueryService.py        # 在线 RAG 流式编排
└── RagStreamEvent.py         # 流式事件
graph_rag/
├── GraphIndexer.py           # 实体关系抽取入库
├── GraphRetriever.py         # 按实体和邻居召回 chunk_id
├── SqliteGraphStore.py       # 单机图谱存储
└── OpenAICompatibleGraphExtractor.py # LLM 实体关系抽取
```

## 生产注意事项

- API 依赖企业网关完成认证并注入可信身份头；必须配置 ACL 代理令牌并启用 TLS。
- 网关必须删除客户端传入的 `X-Tenant-Id`、`X-User-Id`、`X-User-Roles`、
  `X-User-Groups`、`X-User-Clearance` 和 `X-Rag-Proxy-Token`，再写入认证结果。
- 使用本地 bge-m3 和 GPU 时，每个 uvicorn worker 都会加载一份模型；按显存
  控制 worker 数量。Embedding 也可切换到独立的企业 OpenAI 兼容服务。
- SentenceTransformer 首次查询会加载模型；生产环境建议预热模型缓存。
- `API_CORS_ORIGINS` 为空时不启用 CORS。多个来源用逗号分隔，禁止长期使用
  `*` 开放生产接口。
- 当前只提供单轮问答。多轮会话可后续在 `PromptBuilder` 和请求模型中增加
  `messages/history`，无需改动检索与 SSE 协议。
