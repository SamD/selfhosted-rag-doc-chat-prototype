import re

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators


class MarkdownSplitter(FlowFileTransform):
    class ProcessorDetails:
        version = '1.0.0'
        description = 'Performs semantic hierarchical splitting of Markdown content with Zero-Loss sub-splitting logic.'
        tags = ['markdown', 'rag', 'split', 'text', 'semantic']
        dependencies = ['langchain-text-splitters', 'transformers', 'pyyaml', 'mmh3']

    # --- PROPERTY DEFINITIONS ---
    EMBEDDING_MODEL = PropertyDescriptor(
        name="Embedding Model Path",
        description="Path to the local e5-large-v2 model or HuggingFace ID for tokenization.",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR]
    )
    
    MAX_TOKENS = PropertyDescriptor(
        name="Max Tokens",
        description="Strict maximum token limit per chunk.",
        required=True,
        default_value="512",
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR]
    )

    SAFE_BUDGET = PropertyDescriptor(
        name="Safe Budget",
        description="Target token count for initial splitting to allow for prefixes.",
        required=True,
        default_value="450",
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = None

    def getPropertyDescriptors(self):
        return [self.EMBEDDING_MODEL, self.MAX_TOKENS, self.SAFE_BUDGET]

    def transform(self, context, flowfile):
        import yaml
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from transformers import AutoTokenizer

        # 1. Initialization (Lazy load tokenizer)
        model_path = context.getProperty(self.EMBEDDING_MODEL).getValue()
        max_tokens = int(context.getProperty(self.MAX_TOKENS).getValue())
        safe_budget = int(context.getProperty(self.SAFE_BUDGET).getValue())
        
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Read content and metadata
        content = flowfile.read().decode('utf-8')
        trace_id = flowfile.getAttribute('trace_id') or flowfile.getAttribute('uuid')
        doc_id = flowfile.getAttribute('document_id') or "DOC_UNKNOWN"
        
        # 3. Separate YAML header
        header_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if header_match:
            try:
                file_metadata = yaml.safe_load(header_match.group(1))
            except Exception:
                file_metadata = {}
            content_body = content[header_match.end() :]
        else:
            file_metadata = {}
            content_body = content

        # 4. Hierarchical Splitting
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("### [INTERNAL_PAGE_", "Internal_Page"),
            ("###", "Header_3"),
        ]
        
        # Calculate prefix overhead
        enrichment_prefix = f"passage: [{doc_id}] "
        prefix_len = len(self.tokenizer.encode(enrichment_prefix, add_special_tokens=True))
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(content_body)

        def token_len(text):
            return prefix_len + len(self.tokenizer.encode(text, add_special_tokens=False))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=safe_budget - prefix_len,
            chunk_overlap=50,
            length_function=token_len,
            separators=["\n\n", "\n", " ", ""],
        )

        final_chunks = text_splitter.split_documents(sections)

        # 5. Zero-Loss iterative sliding window for overflow
        results = []
        output_chunks = []
        
        for idx, chunk in enumerate(final_chunks):
            current_page = 1
            for k, v in chunk.metadata.items():
                if "Internal_Page" in k or "[INTERNAL_PAGE_" in str(v):
                    page_match = re.search(r"(\d+)", str(v))
                    if page_match:
                        current_page = int(page_match.group(1))

            chunk_text = chunk.page_content
            full_encoded = self.tokenizer.encode(f"{enrichment_prefix}{chunk_text}", add_special_tokens=True)

            if len(full_encoded) <= max_tokens:
                output_chunks.append((chunk_text, current_page))
            else:
                # Sub-split to prevent loss
                content_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
                available_budget = max_tokens - prefix_len - 4 # Safety margin
                
                start_tok = 0
                while start_tok < len(content_tokens):
                    end_tok = min(start_tok + available_budget, len(content_tokens))
                    sub_text = self.tokenizer.decode(content_tokens[start_tok:end_tok], skip_special_tokens=True).strip()
                    if sub_text:
                        output_chunks.append((sub_text, current_page))
                    start_tok = end_tok

        # 6. Generate FlowFile results with fragment attributes
        total_count = len(output_chunks)
        for i, (text, page) in enumerate(output_chunks):
            attributes = {
                "fragment.identifier": trace_id,
                "fragment.index": str(i),
                "fragment.count": str(total_count),
                "page": str(page),
                "document_id": doc_id,
                "chunk_index": str(i),
                "mime.type": "text/plain"
            }
            # Merge with original YAML metadata
            if isinstance(file_metadata, dict):
                for k, v in file_metadata.items():
                    attributes[f"meta.{k}"] = str(v)

            results.append(FlowFileTransformResult(
                relationship="success",
                content=text,
                attributes=attributes
            ))

        return results
