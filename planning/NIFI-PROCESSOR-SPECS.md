# NiFi Configuration & Implementation Guide

This guide provides the exact settings and code needed to deploy the RAG-NiFi pipeline.

## 1. Directory Structure
```text
/
├── nifi/
│   └── python/
│       └── extensions/
│           └── MarkdownSplitter.py  <-- Custom Processor
```

## 2. Phase 1-3: Native Processor Settings

| Processor | Property | Value |
| :--- | :--- | :--- |
| **ListFile** | Input Directory | `/staging` |
| | File Filter | `[^\.].*` |
| **FetchFile** | File to Fetch | `${absolute.path}/${filename}` |
| | Completion Strategy | `Move File` (to `/preprocessing`) |
| **UpdateAttribute** | `trace_id` | `${uuid}` |
| | `document_id` | `${filename:hash('sha256'):substring(0,8)}` |
| **RouteOnAttribute** | `media` | `${filename:matches('(?i).*\.(mp3\|mp4\|wav)')}` |
| | `doc` | `${filename:matches('(?i).*\.(pdf\|html\|txt)')}` |
| **InvokeHTTP** (LXC) | HTTP Method | `POST` |
| | Remote URL | `http://<LXC_IP>:8000/<endpoint>` |
| | Content-Type | `application/octet-stream` |

## 3. Phase 4: Custom Python Processor (`MarkdownSplitter`)
- **API**: NiFi 2.x `FlowFileTransform`
- **Dependencies**: `transformers`, `langchain-text-splitters`, `pyyaml`, `mmh3` (handled automatically by NiFi).
- **Required Properties**:
    - `Embedding Model Path`: Path to your local `e5-large-v2`.
    - `Max Tokens`: `512`
    - `Safe Budget`: `450`

## 4. Phase 5: Vector Storage (`PutQdrant`)
- **NiFi 2.0 Feature**: Use the native `PutQdrant` processor.
- **Qdrant Collection**: `vector_base_collection`
- **Vector**: `${vector}` (extracted from embedding LXC response).
- **Metadata**: Configure `Metadata to Include` to capture `trace_id`, `page`, and `source_file`.

---

## 🚀 How to deploy the Python Processor
1. Copy `nifi/python/extensions/MarkdownSplitter.py` to your NiFi host.
2. In `nifi.properties`, ensure `nifi.python.extensions.directories` includes the path to the extensions folder.
3. Restart NiFi. The `MarkdownSplitter` will appear in the "Add Processor" dialog.
