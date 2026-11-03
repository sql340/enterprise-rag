# enterprise-rag
企业级RAG落地

当前已包含数据读取、文本分块、Embedding 和 Qdrant 向量入库能力。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d qdrant
```

完整的数据处理、分块、向量化和检索示例见
[`data_processing/DATA_PROCESSING_README.md`](data_processing/DATA_PROCESSING_README.md)。
