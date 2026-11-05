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

### 下一步：构建清单

建议增加 `ingestion_manifest` 表，作为构建状态和人工质量记录的依据：

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

### 人工质量闭环

- 支持按文档、页面、知识块标记 `good`、`needs_rework`、`blocked`。
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

文档 ACL 存在关系库，写入时把可过滤字段冗余到 Qdrant payload：

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

### API 与审计

- 网关注入签名身份头或 JWT，服务端校验后生成 `Principal`。
- 每次查询记录：用户、租户、问题摘要、召回文档 ID、过滤原因、模型、耗时和错误码。
- 不在日志中记录完整问题正文和知识正文，避免敏感信息泄漏。
- 知识管理接口单独要求 `knowledge_admin` 权限。

### 推荐的落地顺序

1. 先做单租户 + 密级 + 角色/用户组白名单。
2. 再做多租户硬隔离和 ACL 变更同步。
3. 最后接企业 SSO/OIDC 与目录服务同步用户组。

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

## 4. 本轮先落地的两个完善项

1. 增量索引：未变化块跳过、配置变化局部重建、旧块安全清理。
2. 多路检索：dense + Qdrant BM25 文本检索，通过 RRF 融合排序。

权限模型和入库页面按本文设计分阶段实现，避免在权限还未落地时先做一个可绕过 ACL 的上传页面。
