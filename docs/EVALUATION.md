# 离线检索评测

评测模块用于在修改分块、Embedding、多路检索、Graph RAG 或重排后，用固定问题集
做可重复回归。

## 数据集格式

JSON：

```json
{
  "name": "finance-regression-v1",
  "metadata": {"owner": "knowledge-team"},
  "cases": [
    {
      "case_id": "reimbursement-001",
      "question": "差旅结束后多久内提交报销？",
      "expected_chunk_ids": ["chunk-id-a", "chunk-id-b"]
    },
    {
      "case_id": "policy-002",
      "question": "项目立项需要哪些角色审批？",
      "expected_document_ids": ["document-id-x"]
    }
  ]
}
```

JSONL 每行一个 case，字段与 `cases` 数组元素相同。`case_id` 必须唯一，每个用例
至少提供一个期望 chunk 或 document。

## 指标

- `Recall@K`：期望来源中被前 K 个结果覆盖的比例。
- `MRR`：第一个相关结果排名的倒数。
- `nDCG@K`：二元相关性的归一化折损累计增益。
- `Hit Rate`：前 K 个结果是否至少命中一个期望来源。
- `avg_latency_ms`：按模式统计的平均检索耗时。

同一个数据集会依次运行 `dense`、`hybrid`、`graph`、`hybrid_graph` 等模式，
每个模式单独汇总。

## Python 用法

```python
from access_control import Principal
from evaluation import EvaluationDataset, EvaluationReportStore, EvaluationRunner
from api.app import build_query_service
from configs import RagSettings

dataset = EvaluationDataset.from_file("examples/evaluation_cases.example.json")
service = build_query_service(RagSettings.from_env())
principal = Principal(
    tenant_id="tenant-a",
    user_id="eval-runner",
    roles=frozenset({"knowledge_admin"}),
    clearance="confidential",
)
try:
    report = EvaluationRunner(
        service.retriever,
        principal=principal,
    ).run(
        dataset,
        modes=("dense", "hybrid", "hybrid_graph"),
        top_k=10,
    )
    EvaluationReportStore().save(report, ".cache/evaluation/latest.json")
    print(report.summary)
finally:
    import asyncio
    asyncio.run(service.aclose())
```

评测请求同样经过 `Principal` 和 ACL 过滤，因此只能评估当前评测账号有权访问的
知识。不要使用 `RAG_ACL_ENABLED=false` 生成评测结论后直接套用到生产权限环境。

## 建议的回归门槛

示例基线可以设为：

- `hybrid_graph` 的 Recall@10 不低于当前基线。
- MRR 下降不超过 2%。
- 平均延迟增长不超过 30%，除非有明确收益。
- Graph 或 rerank 开启后如果指标下降，保留配置开关并记录实验结论。
