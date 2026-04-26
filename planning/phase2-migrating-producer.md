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

### To transition from fixed-size chunking to Markdown-aware chunking

To transition from fixed-size chunking to Markdown-aware chunking, the producer_worker should utilize the document's structure to maintain semantic integrity. 
Since your input now includes a YAML-style metadata header, the first step is to isolate the metadata from the content to prevent the header from being fragmented or indexed as prose.

1. Separate Metadata from Content
   Use a regular expression or a dedicated library like python-frontmatter to split the file. The metadata should be parsed into a dictionary and passed along with every chunk generated from that document to ensure traceability in your Rerank/Retrieval steps.

2. Implement Markdown Header Chunking
   Instead of arbitrary character counts, use a hierarchical approach. This ensures that a paragraph remains associated with its relevant heading (H1,H2,H3).
   Split by Headings: Use the # markers as primary split points.
   Respect Token Limits: If a single section (e.g., everything under ## Implementation) exceeds your model's context window, fallback to a recursive character splitter within 그 section.
   Metadata Propagation: Each chunk must retain the original ID, Slug, and Source-Type from the header, while updating the Chunk-Index sequentially.

3. Implementation Example
   Using LangChain's MarkdownHeaderTextSplitter is the standard approach for this, as it handles the logic of nested headers automatically.
   To integrate the Markdown chunking into your existing pipeline while maintaining the IngestState and Redis ordering, you need to ensure total_chunks_sent is updated dynamically based on the results of the semantic splitter.

In your previous fixed-size approach, you likely knew the chunk count based on page numbers or character math. With Markdown, the count is determined after the MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter finish their work.

Updated Markdown Producer Logic
This implementation extracts the YAML header, performs hierarchical splitting, and updates the IngestState so the send_sentinel_node receives the correct total_chunks_sent.

#### Example
```python
import json
import re
import yaml
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def process_markdown_content(state: IngestState) -> IngestState:
    """
    Parses Markdown with metadata headers and enqueues chunks.
    """
    if state["status"] == STATUS_FAILED:
        return state

    rel_path = state["rel_path"]
    queue_name = state["queue_name"]
    
    # Assuming the raw markdown is read from full_path or stored in state
    try:
        with open(state["full_path"], 'r', encoding='utf-8') as f:
            raw_markdown = f.read()

        redis_client = get_redis_client()

        # 1. Separate YAML metadata from Markdown body
        # Matches the block between --- and ---
        header_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', raw_markdown, re.DOTALL)
        if header_match:
            header_yaml = header_match.group(1)
            file_metadata = yaml.safe_load(header_yaml)
            content_body = raw_markdown[header_match.end():]
        else:
            file_metadata = {}
            content_body = raw_markdown

        # 2. Define Markdown splitting hierarchy
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(content_body)

        # 3. Apply secondary recursive splitting for large sections
        # Ensures individual chunks fit within embedding context windows
        def token_len(text):
           # this returns the exact number of tokens
           return len(tokenizer.encode(text))
        
        
        """
        The "recursive" nature comes from its list of separators. It attempts to split the text using the most meaningful separator first, checking the resulting length against your chunk_size.
        The default order of operations is:
            Paragraphs (\n\n)
            Lines (\n)
            Sentences (Spaces)
            Characters (Empty string)
        If a paragraph is longer than your chunk_size, it moves down the list to split that specific paragraph by lines. 
        If a line is still too long, it splits by spaces. This ensures that the text is broken down until every segment is below the character limit you defined.
        """

        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=512, # now this represents tokens, not characters
           chunk_overlap=50,
           length_function=token_len,
           separators=["\n\n", "\n", " ", ""]
        )

        """
        **STILL NEED VERIFICATION THE CHUNK REMAINS IN BUDGET**
        
        Why it Might Still Exceed Limits !
        There are two scenarios where a chunk could still potentially cause issues:
            The "Hard" Limit: If the splitter reaches the end of its separator list (individual characters) and a "word" is still somehow larger than the limit, it will force a break, but this is rare in prose.
            Model Special Tokens: Most embedding models add special tokens (like [CLS] or [SEP]). If you set your chunk_size exactly to 512 tokens, the addition of these model-specific tokens may push the final input to 514 or 515, causing a crash or truncation. It is standard practice to set the chunk_size to 500 to leave a buffer for these headers.
        """
        
        final_chunks = text_splitter.split_documents(sections)

        # 4. Enqueue sequence
        total_chunks = len(final_chunks)
        for idx, chunk in enumerate(final_chunks):
            # Combine document-level metadata with section-level headers
            payload = {
                "type": "data",
                "document_id": state.get("document_id"),
                "content": chunk.page_content,
                "metadata": {
                    **file_metadata,        # Original YAML fields
                    **chunk.metadata,       # Extracted Markdown headers
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "rel_path": rel_path
                }
            }
            
            blocking_push_with_backpressure(
                rclient=redis_client,
                queue_name=queue_name,
                entries=[json.dumps(payload)],
                max_queue_length=50000,
                rel_path=rel_path
            )

        # Update state for the next node (send_sentinel_node)
        return {
            **state, 
            "total_chunks_sent": total_chunks, 
            "status": "PROCESSING_COMPLETE"
        }

    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": str(e)}

```

#### Key Changes to Your Workflow
State Updates: The function now populates total_chunks_sent after the splitting is complete. When send_sentinel_node is called next in your graph/pipeline, state["total_chunks_sent"] will accurately reflect the dynamic number of Markdown chunks created.

Metadata Merging: By merging file_metadata and chunk.metadata, you preserve the ID and Slug from your extraction tier while adding the specific Header_1 or Header_2 where that text originated. This is highly effective for RAG because it provides the LLM with the "breadcrumb" path of the document.

Ordering: Since this entire loop runs before returning the updated state, all data type messages are guaranteed to be in Redis before send_sentinel_node pushes the file_end message.

Schema Consistency: I included document_id in the payload to ensure it aligns with your IngestState definition.

#### Key Considerations for your Pipeline
Metadata as Context: When you eventually feed these chunks into your RAG pipeline, prepend the最重要的 metadata (like Slug) to the text content. This gives the LLM explicit context that might be missing from a small text fragment.

Handling Tables: Markdown tables often break during character-based splitting. If your OCR/Extraction tier produces tables, ensure your separators list in the recursive splitter prioritizes \n\n to avoid cutting a table in half.

Schema Consistency: Since your metadata includes Schema-Version: 2026.04.07, ensure your vector database schema matches this so you can filter by Extraction-Tier or Source-Type during retrieval.