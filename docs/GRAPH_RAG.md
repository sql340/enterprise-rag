# Graph RAG 设计与使用

Graph RAG 在原有 RAG 上增加实体关系图谱召回，并与 dense 向量、BM25 文本检索
共同参与 RRF 融合。图谱不是独立问答链路，而是原有 RAG 的第三路召回。

## 架构

```text
离线构建
  Document
    -> Chunk
    -> Embedding -> Qdrant（正文、向量、ACL）
    -> LLM 实体关系抽取 -> SQLite 图谱（实体、关系、chunk_id、tenant_id）

在线查询
  Question
    -> dense 召回 ─────────────────┐
    -> BM25 召回 ──────────────────┼-> RRF 融合
    -> 实体匹配 + 邻居扩展 -> chunk_id ┘
                                      |
                                      v
                         Qdrant 按 chunk_id + ACL 读取正文
                                      |
                                      v
                                  Prompt + LLM
```

图谱只返回候选 `chunk_id`，不直接返回正文。`KnowledgeRetriever` 会使用与 dense
检索相同的 ACL Filter 回查 Qdrant。因此即使用户能在图谱中匹配到实体，没有文档
读取权限时也不会拿到正文，更不会把内容送入 LLM。

## 图谱模型

| 数据 | 说明 |
| --- | --- |
| 实体 | 规范名称、类型、别名，按 tenant 隔离 |
| 关系 | source、relation_type、target、来源 chunk 和 document |
| chunk 实体链接 | 知识块与实体的多对多关系 |
| chunk 构建状态 | `content_hash + extractor_version`，用于增量跳过 |

当前实现使用 SQLite，适合单机、开发和中等规模知识库。`BaseGraphStore` 已抽象，
后续可替换为 Neo4j、NebulaGraph 等图数据库。

## 构建

```python
from access_control import Classification, DocumentAcl
from data_processing.knowledge_indexing import KnowledgeIndexer
from graph_rag import (
    GraphIndexer,
    OpenAICompatibleGraphExtractor,
    SqliteGraphStore,
)

acl = DocumentAcl(
    tenant_id="tenant-a",
    classification=Classification.INTERNAL,
    owner_id="user-1",
    allowed_roles=frozenset({"finance"}),
)

vector_indexer = KnowledgeIndexer(embedding, vector_store)
vector_result = vector_indexer.index_chunks(chunks, acl=acl)

graph_store = SqliteGraphStore(".cache/knowledge_graph.sqlite3")
extractor = OpenAICompatibleGraphExtractor(
    "gpt-4o-mini",
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
)
graph_indexer = GraphIndexer(graph_store, extractor)
graph_result = graph_indexer.index_chunks(chunks, acl=acl)
```

建议使用独立、低成本、低温度的模型执行离线抽取；在线回答仍使用原来的生成模型。
同一 `chunk_id` 必须同时存在于 Qdrant 和图谱中。

## 增量与删除

- 内容不变且 `extractor_version` 不变时，重复构建会跳过抽取。
- 文件内容变化后，先用 `prune_stale_chunks()` 清理旧 chunk 的图谱节点。
- 文档归档或删除时，调用 `delete_by_document()` 同步清理图谱。
- 图谱构建失败会记录在 `GraphIndexingResult.failures`，默认不阻塞其它知识块。

## 检索模式

| 模式 | 行为 |
| --- | --- |
| `dense` | 原始向量召回 |
| `hybrid` | dense + BM25 |
| `graph` | 仅图谱实体召回 |
| `hybrid_graph` | dense + BM25 + graph，默认 |

三路融合使用 RRF。`sources.route_scores` 会标记命中的检索路；图谱路的分数是
实体匹配和邻居扩展分数，不是向量余弦相似度。

## 配置

```bash
RAG_RETRIEVAL_MODE=hybrid_graph
RAG_GRAPH_ENABLED=true
RAG_GRAPH_STORE_PATH=.cache/knowledge_graph.sqlite3
RAG_GRAPH_WEIGHT=0.8
RAG_GRAPH_MAX_HOPS=1
RAG_GRAPH_ENTITY_LIMIT=20
```

- `RAG_GRAPH_WEIGHT`：图谱路在 RRF 中的权重。
- `RAG_GRAPH_MAX_HOPS`：实体邻居扩展跳数，1 表示只使用问题直接命中的实体。
- `RAG_GRAPH_ENTITY_LIMIT`：问题最多匹配的实体数量。

图谱召回依赖 `Principal.tenant_id`，因此在线 ACL 必须启用。没有图谱数据时，
graph 路返回空，系统自动退化为原有 dense/hybrid 行为。

## 当前限制

- 实体链接目前使用规范化名称和别名的子串匹配，后续可增加实体向量或专门链接模型。
- SQLite 适合单机；大规模图谱应替换为分布式图数据库。
- LLM 抽取质量决定图谱质量，建议保存抽取版本并对低质量实体做人工审核。
- 图谱与向量索引是两套写入流程，生产环境应放入同一个后台任务并保证可重试。
