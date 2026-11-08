from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from data_processing.chunker import Chunk

from .BaseGraphExtractor import BaseGraphExtractor
from .GraphEntity import GraphEntity
from .GraphExtraction import GraphExtraction
from .GraphExtractionError import GraphExtractionError
from .GraphRelation import GraphRelation
from ._utils import normalise_text


class OpenAICompatibleGraphExtractor(BaseGraphExtractor):
    """使用 OpenAI 兼容 Chat Completions 抽取实体关系。

    图谱构建是离线批处理，因此这里使用同步 HTTP；在线问答只访问图存储，不再
    调用抽取模型。
    """

    SYSTEM_PROMPT = """你是企业知识图谱抽取器。请从给定知识块中抽取稳定、可复用的实体和关系。
只输出 JSON，不要输出 Markdown 或解释。JSON 结构必须是：
{"entities":[{"name":"实体名称","type":"person|organization|product|project|policy|location|concept","aliases":["别名"]}],"relations":[{"source":"实体名称","relation":"关系类型","target":"实体名称","description":"关系说明"}]}

规则：
1. 只抽取知识块明确支持的实体和关系，不猜测。
2. 实体名称保持简短、规范；同一实体只出现一次。
3. relations 中的 source 和 target 必须出现在 entities 中。
4. relation 使用简短英文蛇形命名，例如 works_for、belongs_to、requires、owned_by。
5. 没有实体或关系时返回 {"entities":[],"relations":[]}。"""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_input_chars: int = 6000,
        max_entities: int = 30,
        max_relations: int = 60,
        client: httpx.Client | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not base_url.strip():
            raise ValueError("base_url 不能为空")
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")
        if max_input_chars <= 0:
            raise ValueError("max_input_chars 必须为正整数")
        if max_entities <= 0 or max_relations <= 0:
            raise ValueError("max_entities 和 max_relations 必须为正整数")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_entities = max_entities
        self.max_relations = max_relations
        self._client = client
        self._owns_client = client is None

    @property
    def extractor_version(self) -> str:
        return f"openai-graph:{self.model_name}:v1"

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def extract(self, chunk: Chunk) -> GraphExtraction:
        content = chunk.content.strip()[: self.max_input_chars]
        if not content:
            return GraphExtraction()
        heading_path = chunk.metadata.get("heading_path")
        context = f"标题路径: {heading_path}\n" if heading_path else ""
        prompt = f"{context}知识块:\n{content}"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = self._get_client()
        try:
            response = client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GraphExtractionError(f"图谱抽取请求失败: {exc}") from exc

        try:
            raw_content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GraphExtractionError("图谱抽取响应缺少 choices[0].message.content") from exc
        if not isinstance(raw_content, str):
            raise GraphExtractionError("图谱抽取响应内容不是字符串")
        return self.parse_response(raw_content)

    def parse_response(self, content: str) -> GraphExtraction:
        data = self._parse_json(content)
        entities: dict[str, GraphEntity] = {}
        for item in _as_list(data.get("entities")):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            aliases = item.get("aliases") or ()
            if isinstance(aliases, str):
                aliases = (aliases,)
            try:
                entity = GraphEntity(
                    name=name,
                    entity_type=str(item.get("type") or "concept"),
                    aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
                )
            except ValueError:
                continue
            existing = entities.get(entity.normalised_name)
            if existing is None:
                entities[entity.normalised_name] = entity
            elif not existing.aliases and entity.aliases:
                entities[entity.normalised_name] = entity
            if len(entities) >= self.max_entities:
                break

        relations: list[GraphRelation] = []
        for item in _as_list(data.get("relations")):
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            if not source or not target:
                continue
            entities.setdefault(normalise_text(source), GraphEntity(source))
            entities.setdefault(normalise_text(target), GraphEntity(target))
            try:
                relation = GraphRelation(
                    source=source,
                    relation_type=str(item.get("relation") or "related_to"),
                    target=target,
                    description=str(item.get("description") or ""),
                )
            except ValueError:
                continue
            relations.append(relation)
            if len(relations) >= self.max_relations:
                break

        return GraphExtraction(
            entities=tuple(entities.values()),
            relations=tuple(relations),
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise GraphExtractionError("图谱抽取结果不是 JSON 对象")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GraphExtractionError(f"图谱抽取 JSON 解析失败: {exc}") from exc
        if not isinstance(parsed, dict):
            raise GraphExtractionError("图谱抽取结果必须是 JSON 对象")
        return parsed

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


__all__ = ["OpenAICompatibleGraphExtractor"]
