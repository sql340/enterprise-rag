# enterprise-rag
企业级RAG落地

当前已包含数据读取、文本分块、Embedding、Qdrant 向量入库、多租户知识权限、
Graph RAG 实体关系召回、知识治理/质量闭环、离线评测、CrossEncoder 重排，
以及基于 FastAPI + SSE 的在线流式问答能力。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d qdrant
```

启动在线查询 API：

```bash
cp .env.example .env
# 在 .env 中设置 RAG_ACL_PROXY_TOKEN，并让网关注入相同值的 X-Rag-Proxy-Token
set -a
source .env
set +a
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

流式查询接口为 `POST /v1/query`，知识入库控制台位于 `/admin`，OpenAPI
文档位于 `/docs`。

完整的数据处理、分块、向量化和检索示例见
[`data_processing/DATA_PROCESSING_README.md`](data_processing/DATA_PROCESSING_README.md)。

在线接口、SSE 事件协议和客户端示例见
[`api/API_README.md`](api/API_README.md)。

知识质量、权限分级和入库控制台的设计方案见
[`docs/KNOWLEDGE_GOVERNANCE.md`](docs/KNOWLEDGE_GOVERNANCE.md)。

知识管理 API、治理库、质量闭环和控制台见
[`docs/KNOWLEDGE_MANAGEMENT.md`](docs/KNOWLEDGE_MANAGEMENT.md)。

固定问题集的 Recall/MRR/nDCG 评测见
[`docs/EVALUATION.md`](docs/EVALUATION.md)。

Graph RAG 的构建、检索模式和 ACL 安全边界见
[`docs/GRAPH_RAG.md`](docs/GRAPH_RAG.md)。

## 知识权限

权限模型采用默认拒绝的 RBAC + ABAC：租户必须一致，用户密级必须覆盖文档密级，
文档必须为 `active`，并且公开文档、owner、用户白名单、角色白名单或用户组白名单
至少满足一项。管理员仍受租户、密级和文档状态限制。

在线查询从可信网关提供的 `X-Tenant-Id`、`X-User-Id`、`X-User-Roles`、
`X-User-Groups`、`X-User-Clearance` 请求头解析身份。启用 ACL 时必须配置
`RAG_ACL_PROXY_TOKEN`，网关必须覆盖而不是透传外部传入的同名身份头。具体接入
方式见 [`api/API_README.md`](api/API_README.md)。

## Graph RAG

默认检索模式为 `hybrid_graph`：dense 向量、Qdrant BM25 和图谱实体召回三路
并行，通过 RRF 融合。图谱检索只返回候选 `chunk_id`，随后仍从 Qdrant 按当前
用户 ACL 读取正文，因此不会绕过知识权限。

图数据需要与向量索引一起构建；未构建图谱时，graph 路为空，系统自动退化为
dense + BM25 的原始 RAG 行为。

## 数据隔离

`tenant_id` 同时进入知识块 ID 哈希、Qdrant payload 过滤、图谱键和治理库查询。
两个租户上传相同路径和相同内容时不会产生相同的 `document_id/chunk_id`，也
不会互相清理旧块。管理接口只使用可信网关注入的租户身份，不能由请求体覆盖。

## 知识质量闭环

文档清单记录内容哈希、解析器版本、分块配置、Embedding 指纹和图谱抽取版本。
人工标记为 `good` 且构建指纹未变化的文档会直接跳过，不再重复 Embedding；
配置或内容变化时重新构建。低质量块会写入 `quality_flags` 和
`quality_score`，超过阈值时文档自动进入 `needs_rework`，可在控制台按质量
状态批量重建。

## 检索优化

- `RAG_RERANK_ENABLED=true` 时，多路召回和 RRF 融合后使用 CrossEncoder 重排。
- 父子块命中后自动从 Qdrant 读取父块作为 Prompt 上下文，父块同样受 ACL 过滤。
- Graph RAG 仍只返回候选 `chunk_id`，正文必须从 Qdrant 按当前用户权限读取。
