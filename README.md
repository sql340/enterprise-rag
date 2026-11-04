# enterprise-rag
企业级RAG落地

当前已包含数据读取、文本分块、Embedding、Qdrant 向量入库，以及基于
FastAPI + SSE 的在线流式问答能力。

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
