from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .BaseEmbedding import BaseEmbedding
from .EmbeddingDependencyError import EmbeddingDependencyError
from .EmbeddingError import EmbeddingError
from ._utils import _normalise_vectors


class SentenceTransformerEmbedding(BaseEmbedding):
    """通过 sentence-transformers 加载本地或 Hugging Face 模型。

    Args:
        model_name: 模型名称或本地路径，默认 ``BAAI/bge-m3``。
        device: ``cpu``、``cuda``、``mps`` 等；``auto`` 或 ``None`` 自动选择。
        batch_size: 单次编码的文本数量。
        normalize_embeddings: 是否归一化向量。使用余弦距离时建议开启。
        query_prefix: 查询前缀，例如 E5 模型需要 ``query: ``。
        document_prefix: 文档前缀，例如 E5 模型需要 ``passage: ``。
        cache_folder: Hugging Face 模型缓存目录。
        local_files_only: 仅使用本地缓存，不联网下载模型。
        trust_remote_code: 是否允许模型仓库中的自定义代码。
        model_kwargs: 传给 ``SentenceTransformer`` 构造函数的额外参数。
        encode_kwargs: 传给 ``SentenceTransformer.encode`` 的额外参数。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        query_prefix: str = "",
        document_prefix: str = "",
        cache_folder: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        model_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
        model: Any = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = dict(model_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        self._model = model
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._load_model()
            self._dimension = self._model_dimension(model)
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "未安装 sentence-transformers，请执行: "
                "pip install sentence-transformers torch"
            ) from exc

        device = None if self.device in (None, "", "auto") else self.device
        options: dict[str, Any] = {
            "device": device,
            "cache_folder": self.cache_folder,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        options.update(self.model_kwargs)
        try:
            self._model = SentenceTransformer(self.model_name, **options)
        except Exception as exc:
            raise EmbeddingError(
                f"加载 Embedding 模型失败 {self.model_name}: {exc}"
            ) from exc
        return self._model

    @staticmethod
    def _model_dimension(model: Any) -> int:
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            value = getter()
            if value is not None:
                return int(value)
        value = getattr(model, "dimension", None)
        if value is not None:
            return int(value)
        raise EmbeddingError(
            "无法获取 Embedding 维度，请调用 embed_documents() 后再读取 dimension"
        )

    def _embed_texts(
        self,
        texts: Sequence[str],
        *,
        query: bool,
    ) -> list[list[float]]:
        model = self._load_model()
        prefix = self.query_prefix if query else self.document_prefix
        values = [f"{prefix}{text}" for text in texts]
        options = dict(self.encode_kwargs)
        options.setdefault("batch_size", self.batch_size)
        options.setdefault("show_progress_bar", self.show_progress_bar)
        options.setdefault("convert_to_numpy", True)
        options["normalize_embeddings"] = self.normalize_embeddings

        try:
            raw_vectors = model.encode(values, **options)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding 模型编码失败 {self.model_name}: {exc}"
            ) from exc

        vectors = _normalise_vectors(
            raw_vectors,
            expected_count=len(values),
            dimension=self._dimension,
        )
        if self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors


__all__ = ["SentenceTransformerEmbedding"]
