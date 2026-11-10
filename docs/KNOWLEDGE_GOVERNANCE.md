# 知识质量、权限与入库页面设计

本文记录三个后续模块的设计约束，作为实现前的统一方案：稳定知识构建、知识权限分级、知识入库控制台。

## 1. 高质量稳定构建

### 目标

- 同样的文件、分块配置和 Embedding 模型得到完全一致的知识块 ID。
- 没变化的好块不重复 Embedding、不重复写入。
- 只重建受影响的块，不能因为局部优化把整库重刷。
- 分块配置、Embedding 配置、解析器版本变化时，旧块可被识别并清理。
- 构建失败不能误删上一版可用数据。

### 当前实现

- `document_id` 由规范化后的来源和正文哈希计算。
- `chunk_id` 由文档、策略、分块配置哈希、字符区间、正文哈希和父块 ID 计算。
- `KnowledgeIndexer` 默认跳过同一 Embedding 模型下 `chunk_id` 和 `content_hash` 均未变化的块。
- 分块配置变化会产生新 `chunk_id`，新块写入后再清理同一来源的旧块。
- `force_reindex=True` 可以强制重建；Embedding 模型变化会自动触发重建。

### 构建清单（已实现）

`knowledge_documents`、`knowledge_jobs` 和 `knowledge_audit_events` 三张
SQLite 表承担构建清单、任务状态和审计记录：

| 字段 | 说明 |
| --- | --- |
| `source_uri` | 稳定的逻辑来源标识，不用本机绝对路径 |
| `content_hash` | 文件内容哈希 |
| `parser_version` | 解析器版本 |
| `chunk_config_hash` | 分块配置哈希 |
| `embedding_fingerprint` | 模型、维度、前缀、归一化配置的综合指纹 |
| `status` | pending/running/succeeded/failed/skipped |
| `quality_status` | unreviewed/good/needs_rework/blocked |
| `quality_note` | 人工优化说明 |
| `attempt_count` | 失败重试次数 |
| `updated_at` | 最近一次状态变更时间 |

构建流程：

1. 扫描文件并计算 `content_hash`。
2. 如果解析器、分块配置、Embedding 指纹都没有变化，直接跳过整个文件。
3. 只对待重建文件执行读取和分块。
4. 对比现有 `chunk_id`，只对新块或变化块调用 Embedding。
5. 全部写入成功后再清理旧块；失败时保留上一版。
6. 保存构建结果和处理报告。

### 人工质量闭环（已实现）

- 支持按文档标记 `good`、`needs_rework`、`blocked`；低质量块由
  `ChunkQualityAnalyzer` 自动写入块级质量标记。
- 优化分块参数或解析策略后，只重建标记为 `needs_rework` 的文档。
- 人工审核通过的块在配置未变化时永远跳过，除非显式 `force_reindex`。
- 后续评测使用固定问题集，比较优化前后 Recall@K、MRR 和答案引用准确率。

## 2. 知识权限分级

### 权限模型

采用 RBAC + ABAC 的混合模型，默认拒绝：

- 主体：`user_id`、`tenant_id`、角色、用户组、密级 clearance。
- 资源：文档及其知识块、租户、密级 classification、允许角色、允许用户组、允许用户。
- 角色：`platform_admin`、`knowledge_admin`、`editor`、`viewer`。
- 密级：`public`、`internal`、`confidential`、`secret`，映射为整数 0-3。

用户可见一个知识点必须同时满足：

1. `tenant_id` 完全一致。
2. 用户 clearance 大于等于知识密级。
3. 文档状态为 active。
4. 用户属于允许的角色、用户组或用户白名单之一；未配置白名单时默认仅 owner 和 knowledge_admin 可见。

### 存储设计

文档 ACL 已持久化到治理库，并同步冗余到 Qdrant payload；管理接口可在线更新：

| payload 字段 | 用途 |
| --- | --- |
| `tenant_id` | 租户硬隔离 |
| `classification` | 数字密级，支持 Range 过滤 |
| `allowed_roles` | 角色白名单 |
| `allowed_groups` | 用户组白名单 |
| `allowed_users` | 用户白名单 |
| `owner_id` | 所有者 |
| `acl_version` | 权限版本，用于排查和更新 |
| `document_status` | active/archived/deleted |

知识块继承文档 ACL。权限变更时按 `document_id` 批量更新 payload，并递增 `acl_version`。删除文档时先标记 archived，再从检索过滤中排除，最后异步清理向量。

### 检索强制约束

- `tenant_id` 和用户身份只能从认证上下文获取，永远不能由请求 body 覆盖。
- 客户端 `filters` 只能作为附加过滤条件，必须与服务端 ACL Filter 使用 `must` 合并。
- `QueryRequest` 不应允许调用方传入 `tenant_id`、`acl_version` 或任意 ACL 字段。
- 检索、sources 事件、Prompt 拼接三个位置都必须使用同一份已授权结果。
- 对无权限结果不返回内容、不返回标题，也不向 LLM 发送。

### 当前实现

- `access_control` 模块提供 `Classification`、`Principal`、`DocumentAcl`、
  `PrincipalResolver` 和 `QdrantAclFilter`。
- `DocumentAcl.allows()` 与 Qdrant Filter 使用同一套语义：租户、密级、
  active 状态是硬条件；非管理员还需要 owner、显式用户、角色、用户组或 public
  密级中的一项。
- `KnowledgeIndexer` 把文档 ACL 写入每个知识块 payload；ACL 变更可通过
  `update_acl_by_document()` 原地更新，不重算向量。
- `RAG_ACL_ENABLED=true` 时，查询接口从可信身份头解析 `Principal`。未配置
  `RAG_ACL_PROXY_TOKEN` 时查询返回 503，避免信任客户端伪造的身份头。
- 请求 body 中的 ACL 字段会被拒绝；客户端普通过滤条件与服务端 ACL 以 AND
  关系合并。
- ACL 启用前入库且没有 ACL payload 的旧知识块默认不可见，需要迁移后重建。

### API 与审计

- 网关注入签名身份头或 JWT，服务端校验后生成 `Principal`。
- 每次查询记录：用户、租户、问题摘要、召回文档 ID、过滤原因、模型、耗时和错误码。
- 不在日志中记录完整问题正文和知识正文，避免敏感信息泄漏。
- 知识管理接口单独要求 `knowledge_admin` 权限。

### 落地状态

1. 单租户 + 密级 + 角色/用户组白名单：已完成。
2. 多租户硬隔离和 ACL 变更同步：已完成检索与 payload 更新的第一版。
3. ACL 关系清单、管理 API 和审计日志：已完成。
4. 企业 SSO/OIDC：仍由可信网关负责，本服务只消费网关认证后的身份头。

## 3. 知识入库前端

### 页面范围

第一阶段页面只承接管理动作，不把重型解析放进 HTTP 请求线程：

- 上传文件或导入目录。
- 选择分块策略、chunk size、overlap、父子块参数。
- 设置文档密级、允许角色和用户组。
- 解析预览：页数、表格、OCR 文本、空内容告警。
- 分块预览：块数量、长度分布、标题路径和父子关系。
- 提交构建任务，查看进度、失败原因和重试。
- 查看已入库文档、索引版本、质量状态和删除/重建操作。

### 服务拆分

```text
Browser
  -> Ingestion API (auth + job metadata)
  -> Job queue (Redis/RQ, Celery, Dramatiq 或云队列)
  -> Ingestion worker (parse + chunk + embed + Qdrant)
  -> Job status / progress store
```

第一批接口建议：

| 接口 | 说明 |
| --- | --- |
| `POST /v1/admin/documents/upload` | 上传文件并创建待处理文档 |
| `GET /v1/admin/documents` | 文档列表、状态、质量状态 |
| `GET /v1/admin/documents/{id}` | 文档详情和版本信息 |
| `POST /v1/admin/documents/{id}/preview` | 解析与分块预览 |
| `POST /v1/admin/documents/{id}/jobs` | 创建构建任务 |
| `GET /v1/admin/jobs/{job_id}` | 查询任务状态 |
| `POST /v1/admin/jobs/{job_id}/retry` | 重试失败任务 |
| `PATCH /v1/admin/documents/{id}/acl` | 更新权限 |
| `DELETE /v1/admin/documents/{id}` | 归档并异步删除向量 |

### 建议的第一版边界

- 身份先复用网关提供的 header，只在本地开发使用 mock principal。
- 任务队列先使用单进程后台 worker + SQLite job 表，后续替换成 Redis。
- 上传文件保存到本地受控目录，使用随机 ID 作为落盘名，不使用用户原始路径。
- 页面使用 FastAPI 静态 HTML + 原生 JS，避免为了控制台引入新的前端框架。
- 所有写操作都必须校验 `knowledge_admin`，普通查询用户不能访问管理接口。

## 4. 当前完成情况

1. 增量索引：未变化块跳过、配置变化局部重建、旧块安全清理。
2. 多路检索：dense + Qdrant BM25 文本检索，通过 RRF 融合排序。
3. 知识权限：租户、密级、owner、用户、角色、用户组和文档状态过滤已接入
   在线检索链路。
4. Graph RAG：实体关系图谱作为第三路召回，与 dense/BM25 通过 RRF 融合；图谱
   只返回 chunk_id，正文仍从 Qdrant 按 ACL 读取。

已实现：ACL 关系清单、知识管理写接口、审计日志、入库控制台、质量指纹跳过、
质量状态重建、固定问题集评测、CrossEncoder 重排和父块上下文扩展。

当前后台任务使用 SQLite 任务表 + 原子领取/租约/心跳，支持进程重启恢复和
共享同一治理库的多实例 worker。所有实例必须使用同一个 `RAG_GOVERNANCE_DB_PATH`
存储卷；大规模或跨地域部署仍建议替换为 Redis/RQ、Celery 或云队列。

服务启动时自动创建缺失的 Qdrant collection，并补齐 ACL payload 和 BM25 文本
索引；`GET /ready` 在索引或 worker 未就绪时返回 503。
