# Knovera - RAG Assistant

This repository contains the foundation for an internal RAG assistant.

## Setup Instructions

To run this project on a fresh machine, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   - Windows: `.\.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to a new file named `.env`:
     - Windows: `copy .env.example .env`
     - Mac/Linux: `cp .env.example .env`
   - Open `.env` and fill in your actual API keys. (Note: `.env` is ignored by Git, so your secrets are safe.)

5. **Run the application**:
   - With the environment activated and keys in place, you can run:
     - Core API client: `python rag_client.py`
     - Prompt comparison test suite: `python prompt_comparison.py`
     - Token counter & cost estimator: `python token_estimation.py`
      - Document loading & multi-format intake demo: `python document_loading_demo.py`
      - Text extraction & cleaning pipeline demo: `python text_cleaning_demo.py`

## Prompt Construction & Roles (Assignment 3.13)

- **System Prompt**: Template stored in [`prompts/system_prompt.txt`](file:///d:/Project/Knovera/prompts/system_prompt.txt).
- **Prompt Notes & Analysis**: See [`prompts/prompt_notes.md`](file:///d:/Project/Knovera/prompts/prompt_notes.md) for a detailed breakdown of system vs user roles, scope, constraints, and side-by-side prompt variation comparison.
- **Output Logs**: Generated outputs from `prompt_comparison.py` are saved in [`outputs/prompt_comparison_output.txt`](file:///d:/Project/Knovera/outputs/prompt_comparison_output.txt).

## Tokens, Tokenization & Cost Estimation (Assignment 3.14)

- **Token Estimation Script**: [`token_estimation.py`](file:///d:/Project/Knovera/token_estimation.py) calculates token counts using `tiktoken` (`cl100k_base`), computes input vs output billing costs, and models corpus-scale token economics.
- **Analysis & Notes**: Comprehensive notes on token definitions, context limits, non-linear BPE tokenization, and video walkthrough outline are available in [`prompts/token_estimation_notes.md`](file:///d:/Project/Knovera/prompts/token_estimation_notes.md).
- **Output Results**: Full printed counts and cost estimations are saved to [`outputs/token_estimation_output.txt`](file:///d:/Project/Knovera/outputs/token_estimation_output.txt).

## Document Loading & Multi-Format Intake (Assignment 3.19)

- **Core Loader Engine**: Implemented in [`src/document_loader.py`](file:///d:/Project/Knovera/src/document_loader.py), supporting multi-format text extraction (PDF, TXT, HTML, MD), graceful failure handling for missing or unreadable/corrupt files, source metadata preservation, and intake stats.
- **Interactive Demo**: Run [`document_loading_demo.py`](file:///d:/Project/Knovera/document_loading_demo.py) to ingest the sample corpus in [`data/`](file:///d:/Project/Knovera/data/) and print length + text previews for each file.
- **Sample Corpus**: Includes valid multi-format files (`.pdf`, `.txt`, `.html`, `.md`) alongside intentionally corrupted (`corrupt_file.pdf`) and unsupported (`unsupported_file.xyz`) test files.
- **Technical Notes & Video Script**: Comprehensive concepts breakdown, PDF vs text challenges, source identity tracking, error handling, and 3-5 min video script outline are available in [`prompts/document_loading_notes.md`](file:///d:/Project/Knovera/prompts/document_loading_notes.md).
- **Output Results**: Full intake log and statistics saved to [`outputs/document_loading_output.txt`](file:///d:/Project/Knovera/outputs/document_loading_output.txt).

## Text Extraction & Cleaning Pipeline (Assignment 3.20)

- **Core Cleaning Engine**: Implemented in [`src/text_cleaner.py`](file:///d:/Project/Knovera/src/text_cleaner.py), providing a 5-stage transformation pipeline:
  1. Line ending normalization (`\r\n`, `\r` $\rightarrow$ `\n`).
  2. Unicode (NFKC) normalization & mojibake resolution (`â€™` $\rightarrow$ `'`, `â€œ`/`â€` $\rightarrow$ `"`, `â€”` $\rightarrow$ `—`, non-breaking spaces).
  3. Broken mid-word line-wrap repair / de-hyphenation (`transfor-\nmation` $\rightarrow$ `transformation`), while preserving intentional inline compound hyphens (`state-of-the-art`).
  4. Repetitive boilerplate, page numbering (`Page \d+ of \d+`), header/footer, and navigation breadcrumb stripping.
  5. Whitespace collapsing and runaway newline reduction while preserving code indentation and list structure.
- **Interactive Demo & Before/After Proof**: Run [`text_cleaning_demo.py`](file:///d:/Project/Knovera/text_cleaning_demo.py) to clean the document corpus uniformly, inspect side-by-side snippet diffs, and view character count compression metrics.
- **Technical Notes & Video Script**: Detailed guide on RAG noise hazards, vector retrieval degradation, over-cleaning avoidance, edge cases, and 3-5 min video script outline are available in [`prompts/text_cleaning_notes.md`](file:///d:/Project/Knovera/prompts/text_cleaning_notes.md).
- **Output Results**: Execution report and before/after evidence saved to [`outputs/text_cleaning_output.txt`](file:///d:/Project/Knovera/outputs/text_cleaning_output.txt).

## Chunk Metadata & Source Tracking (Assignment 3.22)

- **Metadata Tagging Engine**: Implemented in [`src/chunk_tagger.py`](file:///d:/Project/Knovera/src/chunk_tagger.py), pairing raw text chunks with standardized 10-key metadata dictionaries (`source`, `chunk_index`, `char_start`, `char_end`, `char_length`, `file_type`, `doc_title`, `section_heading`, `page_number`, `effective_date`) across all document formats.
- **Source Tracer & Citation Formatter**: Implemented in [`src/source_tracer.py`](file:///d:/Project/Knovera/src/source_tracer.py), providing exact reverse lookup (`char_start` to `char_end`), line-number calculation (`L5-L6`), string verification audit (`verified_exact_match: True`), standard RAG citation formatting, and metadata-based retrieval filtering.
- **Interactive Demo**: Run `python chunk_metadata_demo.py` to tag multi-format corpus documents, verify metadata schema consistency, run source tracebacks, format citation strings, and filter chunks.
- **Unit Test Suite**: Run `python -m unittest test_chunk_metadata.py` to execute automated test cases covering schema consistency, traceback integrity, citation styles, and retrieval filtering.
- **Technical Notes & Video Script**: Comprehensive breakdown of why chunk metadata powers citations and retrieval filtering, task solutions, and 3-5 min video walkthrough script outline available in [`prompts/chunk_metadata_notes.md`](file:///d:/Project/Knovera/prompts/chunk_metadata_notes.md).
- **Sample Outputs & Logs**: Tagged chunk samples exported to [`sample_tagged_chunks.json`](file:///d:/Project/Knovera/sample_tagged_chunks.json) and execution audit logs saved to [`outputs/chunk_metadata_output.txt`](file:///d:/Project/Knovera/outputs/chunk_metadata_output.txt).

## Clean-Run Proof

The setup instructions have been tested on a fresh setup and everything runs cleanly without any conflicts or missing dependencies.