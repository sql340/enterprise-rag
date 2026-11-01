
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
