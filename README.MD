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

## Token-Aware Chunk Sizing & Overlap (Assignment 3.23)

- **Core Token Chunker Engine**: Implemented in [`src/token_chunker.py`](file:///d:/Project/Knovera/src/token_chunker.py), sizing chunks strictly by token count using `tiktoken` (`cl100k_base`), implementing controlled sliding-window overlap (`step = size - overlap`), and tracking token spans and character lengths.
- **Interactive Demo & Empirical Proof**: Run `python token_chunking_demo.py` to:
  1. Compare token-based sizing vs character-based sizing across dense JSON vs standard prose.
  2. Inspect sliding-window overlap spans across multi-chunk documents.
  3. View live side-by-side evidence showing that **60-token overlap** preserves critical boundary facts that are cut in half by 0-overlap chunking.
  4. Review context window budget math ($4 \times 400 = 1,600$ retrieved tokens) and cost vs context trade-off tables.
  5. Process corpus documents and export sample chunks.
- **Unit Test Suite**: Run `python -m unittest test_token_chunker.py` to verify token sizing enforcement, boundary preservation, overlap stepping, and metadata preservation.
- **Technical Justification Report**: Full parameter justification, model budget calculations, cost-redundancy study, and 3–5 min video script are documented in [`TOKEN_CHUNKING_REPORT.md`](file:///d:/Project/Knovera/TOKEN_CHUNKING_REPORT.md).
- **Sample Outputs & Logs**: Structured chunk exports saved to [`sample_token_chunks.json`](file:///d:/Project/Knovera/sample_token_chunks.json) and execution logs saved to [`outputs/token_chunking_output.txt`](file:///d:/Project/Knovera/outputs/token_chunking_output.txt).

## Embeddings Fundamentals & Vector Representation (Assignment 3.25)

- **Core Embedding Generator**: Implemented in [`src/embedding_generator.py`](file:///d:/Project/Knovera/src/embedding_generator.py), supporting OpenAI / OpenRouter API embeddings (`text-embedding-3-small`, 1536 dimensions) alongside a deterministic offline pseudo-semantic vector engine for zero-dependency test reliability.
- **Vector Similarity Engine**: Implemented `cosine_similarity(a, b)` using `numpy.dot(a, b) / (norm(a) * norm(b))` with numerical precision handling.
- **Interactive Demo & Empirical Proof**: Run `py embedding_demo.py` to:
  1. Generate 1536-dimensional embeddings for sample texts spanning Security/Auth vs Dining/Food domains.
  2. Verify vector dimension uniformity across all sample texts.
  3. Compute pairwise Cosine Similarity matrix and prove that semantically similar pairs (*"reset password"* vs *"recover access login"*: `0.9970`) score significantly higher than dissimilar cross-domain pairs (*"reset password"* vs *"cafeteria menu"*: `-0.0027`).
  4. View plain-English explanations of high-dimensional semantic spaces and why vector embeddings enable semantic search in RAG.
  5. Export demonstration logs and JSON vector previews.
- **Unit Test Suite**: Run `py -m unittest test_embedding_generator.py` to execute automated test cases covering dimension uniformity, cosine similarity identities/orthogonality, and semantic distance ranking assertions.
- **Technical Report & Video Script**: Full task solutions, mathematical formulations, and 3-5 min video walkthrough script available in [`EMBEDDINGS_REPORT.md`](file:///d:/Project/Knovera/EMBEDDINGS_REPORT.md) and [`prompts/embedding_notes.md`](file:///d:/Project/Knovera/prompts/embedding_notes.md).
- **Sample Outputs & Logs**: Trimmed vector samples exported to [`sample_embeddings.json`](file:///d:/Project/Knovera/sample_embeddings.json) and execution audit logs saved to [`outputs/embedding_output.txt`](file:///d:/Project/Knovera/outputs/embedding_output.txt).

## Embedding Similarity & Distance Metrics (Assignment 3.27)

- **Multi-Metric Vector Engine**: Implemented in [`src/embedding_generator.py`](file:///d:/Project/Knovera/src/embedding_generator.py), providing vector similarity & distance calculation routines:
  - `cosine_similarity(a, b)`: Cosine similarity score bounded in `[-1.0, 1.0]`.
  - `cosine_distance(a, b)`: Cosine distance `1.0 - cosine_similarity(a, b)` in `[0.0, 2.0]`.
  - `dot_product(a, b)`: Inner product $\mathbf{a} \cdot \mathbf{b}$ (identical to Cosine Similarity for $L_2$-normalized unit vectors).
  - `euclidean_distance(a, b)`: $L_2$ Euclidean distance $\|\mathbf{a} - \mathbf{b}\|_2$.
  - `manhattan_distance(a, b)`: $L_1$ Manhattan distance $\|\mathbf{a} - \mathbf{b}\|_1$.
- **Chunk Ranking Pipeline**: Added `EmbeddingGenerator.rank_chunks()` and `rank_chunks()`, scoring and sorting chunk embeddings against user queries (descending for similarity, ascending for distance), assigning 1-indexed ranks, and returning structured payloads with `most_similar` (Rank #1) and `least_similar` (Rank #N) chunks.
- **Interactive Demonstration**: Run `py similarity_ranking_demo.py` to:
  1. Compare pairwise metrics across sample text embeddings.
  2. Embed sample query (*"How can a learner reset their password?"*) and compare against 5 corpus chunks across 4 domains.
  3. Display top result (Rank #1: `account-guide.md`, score `0.9967`) and bottom result (Rank #5: `rag-architecture.md`, score `-0.0376`).
  4. Verify cross-metric ranking consistency across Cosine, Dot Product, Cosine Distance, and Euclidean distance.
  5. Export audit logs to [`outputs/similarity_ranking_output.txt`](file:///d:/Project/Knovera/outputs/similarity_ranking_output.txt) and JSON payload to [`sample_similarity_rankings.json`](file:///d:/Project/Knovera/sample_similarity_rankings.json).
- **Automated Unit Test Suite**: Run `py -m unittest test_similarity_ranking.py` to verify metric formulas, mathematical identities, chunk sorting order, top/bottom extraction, cross-metric rank equivalency, and edge case handling.
- **Technical Justification Report**: Full task solutions, metric mathematical justification, high similarity score guarantees vs non-guarantees, and 3-5 min video script outline available in [`SIMILARITY_RANKING_REPORT.md`](file:///d:/Project/Knovera/SIMILARITY_RANKING_REPORT.md) and [`prompts/similarity_ranking_notes.md`](file:///d:/Project/Knovera/prompts/similarity_ranking_notes.md).

## Clean-Run Proof

The setup instructions have been tested on a fresh setup and everything runs cleanly without any conflicts or missing dependencies.

