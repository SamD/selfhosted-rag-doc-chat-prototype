## Phase 2, Migrating Producer Worker
Continuing from the introduction of the lighter gateway model which:
* Monitors staging directory for incoming files
* Converts PDF to Markdown (other media types to come later)
* Does this by using GBNF to enforce first batch in the conversion contains valid markdown and metadata format
* Sends OCR pages to ocr_worker
And the changes to ocr_worker:
* No longer adds enrichment
* Just does straight text extraction
* Now uses docling[easyocr] so no need to check with tesseract and then fallback to easyocr since docling handles this

### The next phase is to address the legacy/dead code in the producer_worker
First off, the code that producer used to send to ocr is no longer valid and needs to be removed. 
Only the gateway is now responsible for sending to OCR

Next is then to overall the chunking strategy to be sent to the consumer.

### producer_worker chunking and tokenization stratecy overhaul
Each gateway normalized document must have a metadata header like the following example:
```yaml
---
ID: c3ed0e56-88fc-4574-9f91-eeaa08712550
Slug: scanned-test-1b31e82f
Processed-At: 2026-04-08T23:29:11.791450+00:00
Source-Type: pdf_ocr_raw
Extraction-Tier: 3
Chunk-Index: 0
Schema-Version: 2026.04.07
Raw-Path: /home/samueldoyle/Projects/GitHub/SamD/selfhosted-rag-doc-chat-prototype/Docs/staging/scanned_test.pdf
---
```

To transition from fixed-size chunking to Markdown-aware chunking, the producer_worker should utilize the document's structure to maintain semantic integrity. Since your input now includes a YAML-style metadata header, the first step is to isolate the metadata from the content to prevent the header from being fragmented or indexed as prose.

1. Separate Metadata from Content
   Use a regular expression or a dedicated library like python-frontmatter to split the file. The metadata should be parsed into a dictionary and passed along with every chunk generated from that document to ensure traceability in your Rerank/Retrieval steps.

2. Implement Markdown Header Chunking
   Instead of arbitrary character counts, use a hierarchical approach. This ensures that a paragraph remains associated with its relevant heading (H1,H2,H3).

Split by Headings: Use the # markers as primary split points.

Respect Token Limits: If a single section (e.g., everything under ## Implementation) exceeds your model's context window, fallback to a recursive character splitter within 그 section.

Metadata Propagation: Each chunk must retain the original ID, Slug, and Source-Type from the header, while updating the Chunk-Index sequentially.

3. Implementation Example
   Using LangChain's MarkdownHeaderTextSplitter is the standard approach for this, as it handles the logic of nested headers automatically.

Python
import re
import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def producer_worker_markdown_logic(raw_markdown: str):
# 1. Extract Metadata Header
# Matches content between the first two '---' delimiters
header_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', raw_markdown, re.DOTALL)
if header_match:
metadata = yaml.safe_load(header_match.group(1))
content = raw_markdown[header_match.end():]
else:
metadata = {}
content = raw_markdown

    # 2. Define Header Splitting Rules
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)

    # 3. Recursive splitting for oversized sections
    # Ensures chunks stay within embedding model constraints (e.g., 512 or 1024 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(md_header_splits)

    # 4. Enrich chunks with original metadata
    for i, chunk in enumerate(final_chunks):
        chunk.metadata.update(metadata)
        chunk.metadata["Chunk-Index"] = i
        
    return final_chunks
Key Considerations for your Pipeline
Metadata as Context: When you eventually feed these chunks into your RAG pipeline, prepend the最重要的 metadata (like Slug) to the text content. This gives the LLM explicit context that might be missing from a small text fragment.

Handling Tables: Markdown tables often break during character-based splitting. If your OCR/Extraction tier produces tables, ensure your separators list in the recursive splitter prioritizes \n\n to avoid cutting a table in half.

Schema Consistency: Since your metadata includes Schema-Version: 2026.04.07, ensure your vector database schema matches this so you can filter by Extraction-Tier or Source-Type during retrieval.