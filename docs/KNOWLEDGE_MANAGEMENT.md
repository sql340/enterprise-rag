# 知识管理、质量闭环与租户隔离

知识管理模块把“上传 → 解析 → 分块 → 向量化 → 图谱构建 → 质量复核”串成一条
可审计、可重试的离线链路。入口为 `/admin` 控制台和 `/v1/admin/**` API。

## 模块结构

```text
knowledge_management/
├── KnowledgeDocument.py          # 文档清单与构建指纹
├── IngestionJob.py               # 持久化任务
├── IngestionJobLeaseError.py     # 租约失效异常
├── IngestionJobWorker.py         # 原子领取、续租与重启恢复
├── AuditEvent.py                 # 审计事件
├── KnowledgeGovernanceStore.py   # SQLite 文档/任务/审计存储
├── KnowledgeIngestionService.py  # 入库业务编排
├── KnowledgeAdminService.py      # 管理 API 的授权与适配
├── KnowledgeBuildFingerprint.py  # 内容/解析/分块/模型综合指纹
└── ChunkQualityAnalyzer.py       # 块级质量检查
```

数据表：

- `knowledge_documents`：来源、文件元数据、内容哈希、ACL、分块配置、构建指纹、
  质量状态、索引版本。
- `knowledge_jobs`：pending/running/succeeded/failed、进度、错误和完整构建报告。
- `knowledge_audit_events`：租户、操作者、动作、目标与详情。

所有读取都显式带 `tenant_id`。文档主键跨租户冲突会被拒绝，任务也只允许读取
同租户文档。

## 租户隔离

仅给向量 payload 加 `tenant_id` 不够：两个租户上传相同路径和相同内容时，来源
哈希和分块 ID 可能相同，最终互相覆盖。`TenantChunkScope` 在写入前把
`chunk_id`、`document_id` 和 `parent_id` 重新哈希：

```text
sha256("tenant-scope:v1" + tenant_id + kind + original_id)
```

同步清理 `prune_stale_chunks()`、ACL payload 更新、按文档删除也全部要求
`tenant_id`。图谱的 chunk/document/entity 键同样按租户隔离。

## 稳定构建与跳过策略

文档构建指纹包含：

- 文件来源、类型和内容哈希。
- 解析器版本。
- 分块策略与完整分块参数。
- Embedding 提供者、模型、collection 指纹。
- 图谱抽取器版本。

跳过条件必须同时满足：

1. 文档上次构建成功。
2. 人工质量状态为 `good`。
3. 当前构建指纹与持久化指纹完全一致。
4. 没有显式 `force_reindex=true`。

满足时仍会生成一条 `succeeded / skipped_unchanged` 任务记录，方便审计和客户端
轮询，但不会重新解析、Embedding 或写入向量。

内容或配置变化、Embedding 模型变化、解析器版本变化都会使指纹失效。图谱抽取器
变化会触发任务重建，但向量索引器仍按 `content_hash + embedding_model + ACL`
跳过未变化的块。

## 质量闭环

`ChunkQualityAnalyzer` 对每个块检查：

- 过短或过长。
- 信息密度过低（符号/空白占比过高）。
- 重复字符比例异常。

结果写入 payload：

```json
{
  "quality_flags": ["too_short"],
  "quality_score": 0.75,
  "quality_review_required": true
}
```

低质量块超过阈值时文档自动标记 `needs_rework`，任务报告包含低质量块样本。
人工可在控制台标记 `good`、`needs_rework`、`blocked`；按 `needs_rework`
批量重建时使用 `force_reindex=true`，并继续记录审计。

## 管理 API 与安全边界

管理接口要求：

- `RAG_ACL_PROXY_TOKEN` 已配置，否则返回 503。
- 可信网关提供 `X-Tenant-Id`、`X-User-Id` 等身份头。
- 角色包含 `knowledge_admin` 或 `platform_admin`。
- 管理员仍不能跨租户，也不能操作高于自身 clearance 的文档。

ACL 更新直接写入 Qdrant payload，不重算向量；归档会先切换 ACL 的
`document_status`，查询立即不可见，再决定是否删除向量和图谱。

## 异步任务边界

构建任务持久化在 SQLite，由进程内 `IngestionJobWorker` 轮询领取，不再依赖
FastAPI 请求生命周期内的 `BackgroundTasks`。领取任务使用 `BEGIN IMMEDIATE`
原子事务，并通过 `worker_id + lease_expires_at` 防止多个实例重复处理。

worker 在任务开始、解析、分块、向量化、图谱构建和完成前续租。实例崩溃或重启
后，pending 任务以及租约过期的 running 任务会被新 worker 接管；同一任务累计
尝试超过 `RAG_INGESTION_MAX_ATTEMPTS` 后会标记失败。失去租约的旧 worker 不会
继续覆盖任务状态。

多实例部署必须共享同一个 `RAG_GOVERNANCE_DB_PATH` 和上传目录，并使用支持
SQLite 文件锁的本地/共享块存储。更大规模或跨地域部署仍建议将任务表替换为
Redis/RQ、Celery 或云队列。

失败任务会保留错误信息，可通过 `POST /v1/admin/jobs/{job_id}/retry` 创建新任务。
同一文档同时只允许一个 pending/running 任务。

## 检索索引初始化

服务启动时调用 Qdrant 索引 bootstrap：collection 不存在则按当前 Embedding
维度创建，随后补齐 ACL 过滤字段和 BM25 文本索引。`GET /ready` 会验证索引
状态，未就绪时返回 503，避免 `hybrid_graph` 在无提示的情况下退化为 dense，
或 ACL 过滤缺少索引。关闭自动 bootstrap 时，应由外部发布流程先完成建库。
