# 可观测性第一阶段

当前实现不引入额外的日志系统或 OpenTelemetry Collector，先在应用内提供：

- JSON 结构化日志
- `X-Request-ID` / `X-Trace-ID` 请求关联
- Prometheus `/metrics`
- HTTP、查询、检索、Embedding、LLM、知识构建指标
- Embedding / LLM 依赖健康检查

## 接口

- `GET /health`：进程存活检查，不访问外部依赖。
- `GET /ready`：检查 Qdrant 索引、worker、管理服务和依赖状态。
- `GET /metrics`：Prometheus 指标，建议只暴露给内网监控系统。

`/ready` 的 `status` 有三种值：

- `ready`：索引、worker 和依赖全部正常。
- `degraded`：核心服务可用，但 Embedding / LLM 等依赖不可用。
- `not_ready`：索引或 worker 等核心状态未就绪。

默认情况下外部依赖失败会让 `/ready` 返回 `200 + degraded`，避免所有副本被负载
均衡器同时摘除。要求依赖失败必须返回 503 时，设置：

```bash
RAG_DEPENDENCY_HEALTH_FAIL_READINESS=true
```

## 结构化日志

默认输出 JSON，每行一个日志对象。请求日志包含：

```json
{
  "timestamp": "2026-11-13T10:00:00.000000Z",
  "level": "INFO",
  "logger": "enterprise_rag.access",
  "message": "HTTP 请求完成",
  "request_id": "request-123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "event": "http.request",
  "method": "POST",
  "route": "/v1/query",
  "status": 200,
  "duration_ms": 1250.4
}
```

请求头中的合法 `X-Request-ID` 会被复用，否则自动生成。`traceparent` 中的
W3C trace id 也会被提取，方便后续接入 OpenTelemetry。

默认不记录问题、答案、知识块正文、用户 ID 等业务内容。需要排查时也建议通过
临时采样和脱敏开关控制，不要把完整内容写入长期日志。

应用自身已经输出访问日志，部署时建议关闭 Uvicorn 的重复访问日志：

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080 --no-access-log
```

## 配置

```bash
RAG_LOG_LEVEL=INFO
RAG_LOG_FORMAT=json
RAG_ACCESS_LOG_ENABLED=true
RAG_METRICS_ENABLED=true
RAG_WARMUP_EMBEDDING=true

# 依赖健康检查；结果会缓存，避免每个探针都请求模型服务
RAG_DEPENDENCY_HEALTH_CACHE_SECONDS=15
RAG_DEPENDENCY_HEALTH_TIMEOUT_SECONDS=2
RAG_DEPENDENCY_HEALTH_EXTERNAL_ENABLED=true
RAG_DEPENDENCY_HEALTH_FAIL_READINESS=false

# 网关不提供 /models 时，可显式指定轻量健康检查地址
RAG_DEPENDENCY_HEALTH_LLM_URL=
RAG_DEPENDENCY_HEALTH_EMBEDDING_URL=
```

本地 SentenceTransformer 模型会在启动时预热；设置
`RAG_WARMUP_EMBEDDING=false` 可以关闭预热。OpenAI 兼容 Embedding / LLM 默认
检查 `<base_url>/models`，如果企业网关没有该接口，请配置上面的健康检查 URL。

## 主要指标

| 指标 | 说明 |
| --- | --- |
| `rag_http_requests_total` | HTTP 请求数 |
| `rag_http_request_duration_seconds` | HTTP 请求耗时 |
| `rag_query_requests_total` | 查询结果分布：success/error/no_context/disconnected |
| `rag_query_first_token_seconds` | 从请求开始到首个回答增量 |
| `rag_retrieval_duration_seconds` | 检索总耗时 |
| `rag_retrieval_results_count` | 召回结果数量 |
| `rag_retrieval_empty_total` | 空召回次数 |
| `rag_retrieval_route_duration_seconds` | dense/keyword/graph/rerank 等分路耗时 |
| `rag_embedding_duration_seconds` | Embedding 耗时 |
| `rag_llm_ttft_seconds` | LLM 首 token 时间 |
| `rag_llm_stream_duration_seconds` | LLM 流式生成耗时 |
| `rag_ingestion_queue_depth` | 构建任务队列深度 |
| `rag_ingestion_job_duration_seconds` | 构建任务耗时 |
| `rag_dependency_up` | Embedding / LLM 依赖是否可用 |
| `rag_acl_denials_total` | 鉴权失败次数 |

不要在 Prometheus label 中加入租户 ID、用户 ID、文档 ID、问题文本等高基数或
敏感字段。指标只用于聚合，详细追踪放到后续的 Trace 后端。

## Prometheus 接入

应用已经暴露 `/metrics`，只需让现有 Prometheus 抓取：

```yaml
scrape_configs:
  - job_name: enterprise-rag
    metrics_path: /metrics
    static_configs:
      - targets: ["enterprise-rag:8080"]
```

建议先配置四条告警：

- 5xx 比例持续高于 1%
- 查询/首 token P95 超过基线
- Qdrant 或 worker 未就绪超过 2 分钟
- `pending` 队列持续增长或最老任务超过 15 分钟

第一阶段不要求部署 Grafana、Loki、Tempo 或 OpenTelemetry Collector。等需要
跨服务链路分析时，再在现有 Prometheus/Grafana 基础上增加 OpenTelemetry Trace。
