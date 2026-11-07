# enterprise-rag
企业级RAG落地

当前已包含数据读取、文本分块、Embedding、Qdrant 向量入库、多租户知识权限，
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

流式查询接口为 `POST /v1/query`，OpenAPI 文档位于 `/docs`。

完整的数据处理、分块、向量化和检索示例见
[`data_processing/DATA_PROCESSING_README.md`](data_processing/DATA_PROCESSING_README.md)。

在线接口、SSE 事件协议和客户端示例见
[`api/API_README.md`](api/API_README.md)。

知识质量、权限分级和入库控制台的设计方案见
[`docs/KNOWLEDGE_GOVERNANCE.md`](docs/KNOWLEDGE_GOVERNANCE.md)。

## 知识权限

权限模型采用默认拒绝的 RBAC + ABAC：租户必须一致，用户密级必须覆盖文档密级，
文档必须为 `active`，并且公开文档、owner、用户白名单、角色白名单或用户组白名单
至少满足一项。管理员仍受租户、密级和文档状态限制。

在线查询从可信网关提供的 `X-Tenant-Id`、`X-User-Id`、`X-User-Roles`、
`X-User-Groups`、`X-User-Clearance` 请求头解析身份。启用 ACL 时必须配置
`RAG_ACL_PROXY_TOKEN`，网关必须覆盖而不是透传外部传入的同名身份头。具体接入
方式见 [`api/API_README.md`](api/API_README.md)。
