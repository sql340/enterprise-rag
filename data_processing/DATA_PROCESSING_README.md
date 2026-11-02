
# 数据处理

负责RAG模块离线数据处理部分，包括：数据的读取、分块，向量embedding、索引入库相关
1. 考虑数据处理的稳定性。比如数据重新处理分块，向量索引等，知识没有更新时，跟上一版要一致
2.

## 文件读取

`data_reader.py` 提供统一的 `DataReader`，将不同格式的文件转换为可供分块、
向量化和检索的文本，并保留来源、页码、工作表等元数据。

### 支持格式

| 类别 | 扩展名 | 说明 |
| --- | --- | --- |
| Markdown / 文本 | `.md`, `.markdown`, `.txt` | 自动处理 UTF-8、GB18030 等编码 |
| PDF | `.pdf` | 优先读取文本层；扫描页可调用 OCR |
| Word | `.docx`, `.docm`, `.doc` | `.doc` 需要 LibreOffice 转换 |
| PowerPoint | `.pptx`, `.pptm`, `.ppsx`, `.potx`, `.ppt` | 按页读取文字、表格、备注和内嵌图片 OCR |
| 表格 | `.csv`, `.tsv`, `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.xls`, `.ods` | 按工作表读取并转为 Markdown 表格 |
| 图片 | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp` | 读取图片元数据并执行 OCR |
| 音视频 | `.mp4`, `.mov`, `.avi`, `.mkv`, `.mp3`, `.wav` 等 | 读取媒体信息，按需执行语音转写 |

### 安装依赖

按实际使用的文件类型安装即可，缺失依赖时会抛出带安装提示的
`OptionalDependencyError`：

```bash
pip install pypdf pdfplumber pymupdf python-docx python-pptx openpyxl xlrd
pip install pillow pytesseract faster-whisper
```

此外还需要系统级命令：

- `tesseract`：图片和扫描版 PDF 的 OCR。
- `ffmpeg` / `ffprobe`：音视频信息读取、音频提取和转写。
- `libreoffice` / `soffice`：`.doc`、`.ppt`、`.xls`、`.ods` 的格式转换。

### 基础用法

```python
from data_processing.data_reader import DataReader, read_file

reader = DataReader()

# 一个 PDF 页面、PPT 页面或 Excel 工作表对应一个 Document
documents = reader.read("knowledge/demo.pdf")
for document in documents:
    print(document.metadata["source"])
    print(document.metadata.get("page_number"))
    print(document.content)

# 顶层便捷函数
documents = read_file("knowledge/demo.xlsx", max_table_rows=1000)
```

`read()` 始终返回 `list[Document]`，每个 `Document` 包含：

- `content`：Markdown 或纯文本内容。
- `metadata`：至少包含 `source`、`file_name`、`extension`、`file_type`、
  `mime_type`、`size_bytes`、`modified_at` 和 `char_count`；PDF、PPT、
  Excel 还会包含页码、幻灯片编号或工作表名称。
- `page_content`：`content` 的兼容性别名；也可用 `to_dict()` 转为字典。

### 批量读取与自定义 OCR / ASR

批量读取默认按路径排序，结果稳定，并跳过隐藏文件和未知格式：

```python
reader = DataReader(
    ocr_handler=lambda image_path: my_ocr_service(image_path),
    transcriber=lambda media_path: my_asr_service(media_path),
)

documents = reader.read_directory(
    "knowledge",
    recursive=True,
    extensions={".pdf", ".docx", ".md"},
)

# 单个文件失败时继续处理其它文件
documents = reader.read_directory(
    "knowledge",
    continue_on_error=True,
    on_error=lambda path, error: print(path, error),
)
```

默认 OCR 使用 Tesseract，默认音视频转写使用 faster-whisper（回退到
openai-whisper）。生产环境建议通过 `ocr_handler` 和 `transcriber` 注入企业
已有的 OCR / ASR 服务，以统一鉴权、并发控制和结果格式。

PDF 按页输出，PPT 按幻灯片输出，Excel 按工作表输出；Markdown、Word 和单个
图片则各输出一个 `Document`。

## 文本分块

`chunker.py` 将 `Document` 转换为 `Chunk`。所有策略都支持单文档、多文档和
纯字符串输入，继承原始元数据，并生成可复现的 `chunk_id`。

### 内置策略

| 策略 | 名称 | 说明 | 推荐场景 |
| --- | --- | --- | --- |
| 固定长度 | `fixed` | 按字符或自定义长度单位切分，支持 overlap | 基线、日志、无明显结构的文本 |
| 递归分块 | `recursive` | 按段落、换行、中英文句子、逗号等优先级递归切分 | 通用默认策略 |
| 句子窗口 | `sentence_window` | 按中英文句子合并，可附带前后句窗口 | 问答、政策、新闻 |
| 段落分块 | `paragraph` | 按自然段切分，过长段落自动递归拆分 | 报告、规范、书籍 |
| Markdown 标题 | `markdown_heading` | 按 H1-H6 分块，保留标题路径，跳过代码围栏中的伪标题 | Markdown 知识库 |
| 父子分层 | `parent_child` | 先生成父块，再在父块内生成子块，输出父子关系 | 企业 RAG 检索与上下文扩展 |

所有策略默认按字符计数，并支持 `length_function` 注入 tokenizer 计数函数，
实现 token-aware 分块。

### 基础用法

```python
from data_processing.chunker import Chunker, chunk_documents
from data_processing.data_reader import DataReader

documents = DataReader().read("knowledge/demo.md")

chunker = Chunker(
    strategy="markdown_heading",
    chunk_size=600,
    chunk_overlap=80,
)
chunks = chunker.split(documents)

for chunk in chunks:
    print(chunk.chunk_id)
    print(chunk.metadata["heading_path"])
    print(chunk.content)

# 一次性调用
chunks = chunk_documents(documents, strategy="recursive", chunk_size=600)
```

`chunk_size` 的上限单位由 `length_function` 决定，默认是字符。`chunk_overlap`
未显式传入时取 `min(64, chunk_size // 8)`，显式传入时必须小于
`chunk_size`。`min_chunk_size` 用于尝试合并尾部过小的块。

### Token 计数

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

chunker = Chunker(
    "recursive",
    chunk_size=512,
    chunk_overlap=64,
    length_function=lambda text: len(encoding.encode(text)),
)
chunks = chunker.split(documents)
```

### 句子窗口

```python
chunker = Chunker(
    "sentence_window",
    chunk_size=400,
    chunk_overlap=50,
    sentence_window=2,
)
chunks = chunker.split(documents)

# 正文用于检索；window_content 可用于返回命中句子的上下文
window_text = chunks[0].metadata.get("window_content")
```

### 父子分层

```python
chunker = Chunker(
    "parent_child",
    chunk_size=400,             # 子块大小
    chunk_overlap=50,           # 子块 overlap
    parent_chunk_size=1600,
    parent_chunk_overlap=0,
)
chunks = chunker.split(documents)

children = [chunk for chunk in chunks if chunk.chunk_role == "child"]
parents_by_id = {
    chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_role == "parent"
}

# 检索 child，命中后按 parent_id 取父块
parent = parents_by_id[children[0].parent_id]
```

### Chunk 元数据

每个 `Chunk` 包含：

- `content`、`chunk_id`、`document_id`、`chunk_index`、`char_start`、
  `char_end`、`content_hash`、`parent_id`。
- 原始 `Document` 元数据，例如 `source`、`page_number`、`sheet_name`。
- `chunk_strategy`、`chunk_config_hash`、`chunk_role`、`chunk_role_index`、
  `chunk_role_count`、`chunk_length` 和 `chunk_content_hash`。
- Markdown 的 `heading_path`、`section_title`；句子策略的
  `sentence_start_index`、`sentence_end_index`、`window_content`；段落策略的
  `paragraph_start_index`、`paragraph_end_index`。
- 父子策略的 `parent_index`、`parent_char_start`、`parent_char_end`、
  `child_index`、`child_count`。

`document_id` 由来源路径和规范化后的文档内容计算；`chunk_id` 还包含策略、
分块配置、字符位置、正文哈希和父块 ID。因此同一内容在相同配置下重复处理
会得到稳定 ID，配置变化时 ID 会同步变化，便于判断索引是否需要更新。
