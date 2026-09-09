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

## Retrieval Relevance Tuning (Assignment 3.34)

- **Enterprise Tuning Engine**: Implemented in [`src/retrieval_tuner.py`](file:///d:/Project/Knovera/src/retrieval_tuner.py), managing ground-truth test query pairs (`TestQuery`), candidate retrieval configurations (`RetrievalSetting`), score thresholding (`min_score`), metadata filtering, and hybrid search blending.
- **Relevance Metrics Computation**: Evaluates Hit Rate, Top-1 Hit Rate, Mean Reciprocal Rank (MRR), Average Top Score, and Average Retained Chunks across candidate settings.
- **Quantitative Setting Selection**: Implements automated multi-criteria optimization to select winning retrieval parameters with evidence-backed justification.
- **Interactive Demonstration**: Run `python retrieval_relevance_tuning_demo.py` to:
  1. Index multi-domain grounded test corpus into ChromaDB collection.
  2. Evaluate 5 retrieval configurations (`baseline_k3`, `filtered_k3`, `strict_k5`, `hybrid_k3`, `hybrid_filtered_strict_k3`) across 5 ground-truth test queries.
  3. Display comparative empirical results table (100% Hit Rate for `hybrid_k3` with 46% reduced context token cost).
  4. Export JSON metrics to [`outputs/retrieval_relevance_tuning_results.json`](file:///d:/Project/Knovera/outputs/retrieval_relevance_tuning_results.json) and plain-text execution log to [`outputs/retrieval_relevance_tuning_output.txt`](file:///d:/Project/Knovera/outputs/retrieval_relevance_tuning_output.txt).
- **Automated Pytest Suite**: Run `pytest test_retrieval_relevance_tuning.py -v` to verify dataclass serialization, single/multi setting evaluations, score filtering, hit rate calculations, and best setting selection logic.
- **Comprehensive Technical Report**: Full report, pipeline architecture flowchart, empirical comparison tables, setting selection justification, impact analysis, and video script outline available in [`RETRIEVAL_RELEVANCE_TUNING_REPORT.md`](file:///d:/Project/Knovera/RETRIEVAL_RELEVANCE_TUNING_REPORT.md).

## Retrieval Evaluation & Recall Testing (Assignment 3.36)

- **Enterprise Evaluator Engine**: Implemented in [`src/retrieval_evaluator.py`](file:///d:/Project/Knovera/src/retrieval_evaluator.py), managing labelled query benchmark sets (`LabelledQuery`), computing exact **Recall@k**, **Precision@k**, **F1-Score**, **Mean Reciprocal Rank (MRR)**, and **Hit Rate@k**, and performing automated failure cause inspection (`inspect_failures`).
- **Benchmark Labelled Dataset**: Defined in [`sample_labelled_queries.json`](file:///d:/Project/Knovera/sample_labelled_queries.json), mapping representative user queries to explicit sets of ground-truth relevant chunk IDs across 6 domain categories.
- **Multi-K Trade-Off Analysis**: Evaluates retrieval quality across varying top-$k$ values ($k=1, 3, 5, 10$) to demonstrate coverage expansion vs precision noise trade-offs:
  - $k=1$: Recall@1 = `75.0%`, Precision@1 = `83.3%`, Passed = `4/6`
  - $k=3$: Recall@3 = `91.7%`, Precision@3 = `38.9%`, Passed = `5/6`
  - $k=5$: Recall@5 = `100.0%`, Precision@5 = `26.7%`, Passed = `6/6` (Optimal Recall)
  - $k=10$: Recall@10 = `100.0%`, Precision@10 = `13.3%`, Passed = `6/6`
- **Interactive Demonstration**: Run `py retrieval_evaluation_demo.py` to:
  1. Index benchmark corpus into ChromaDB vector store.
  2. Evaluate single and batch labelled queries.
  3. Perform multi-k evaluation ($k=1, 3, 5, 10$) and display summary tables.
  4. Inspect low-recall queries ($k=1$), diagnose root causes (`TOO_SMALL_K`, `ZERO_HITS_EMBEDDING_MISMATCH`), and output actionable recall optimization roadmaps.
  5. Export JSON metrics to [`outputs/retrieval_evaluation_results.json`](file:///d:/Project/Knovera/outputs/retrieval_evaluation_results.json) and plain-text audit log to [`outputs/retrieval_evaluation_audit.log`](file:///d:/Project/Knovera/outputs/retrieval_evaluation_audit.log).
- **Automated Pytest Suite**: Run `pytest test_retrieval_evaluator.py -v` to verify dataclass serialization, recall/precision metric formulas, multi-k analysis, failure diagnostics, and export functionality.
- **Comprehensive Technical Report**: Full architectural flowcharts, metric equations, empirical result tables, failure diagnostics, and recall optimization roadmap available in [`RETRIEVAL_EVALUATION_RECALL_TESTING_REPORT.md`](file:///d:/Project/Knovera/RETRIEVAL_EVALUATION_RECALL_TESTING_REPORT.md).

## RAG Pipeline Architecture & Flow Design (Assignment 3.37)

- **End-to-End Pipeline Architecture**: Implemented in [`src/rag_pipeline.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/rag_pipeline.py), connecting the stages: `User Query -> Embed Query -> Retrieve Context (Top-K) -> Assemble Context -> Generate Grounded Answer -> Return Answer + Sources`.
- **Separated, Testable Code Stages**:
  1. `embed_query(query)`: Converts user questions into 1536-dimensional normalized embedding vectors.
  2. `retrieve_context(query_vector, k=4)`: Performs top-k vector similarity search in ChromaDB with cosine distance scoring and optional score threshold filtering.
  3. `assemble_context(chunks)`: Structures retrieved evidence chunks into indexed citation blocks (`[index] Source: doc.md | Section: sec\n<text>`).
  4. `generate_answer(query, context)`: Synthesizes a factual, grounded response using structured prompt templates and LLM completion with offline extractive fallback.
  5. `answer_query(query, k=4)`: Coordinates end-to-end execution, tracks per-stage latencies, and enforces strict non-hallucination fallback when retrieval returns no context.
- **Interactive Demonstration**: Run `python rag_pipeline_demo.py` to:
  1. Display complete ASCII RAG pipeline architecture diagram.
  2. Execute step-by-step isolated stages with latency tracking.
  3. Run positive and out-of-domain sample queries demonstrating grounded answering and empty retrieval fallback.
  4. Export full JSON audit logs to [`sample_rag_pipeline_output.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_rag_pipeline_output.json).
- **Automated Test Suite**: Run `python test_rag_pipeline.py` to verify 12 unit and integration test cases across all stages.
- **Comprehensive Technical Report**: Complete architectural breakdowns, stage data flows, mathematical grounding, sample executions, and video walkthrough script available in [`RAG_PIPELINE_ARCHITECTURE_FLOW_REPORT.md`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/RAG_PIPELINE_ARCHITECTURE_FLOW_REPORT.md).

## Context Injection & Prompt Augmentation (Assignment 3.38)

- **Grounded Context Injection Engine**: Implemented in [`src/context_injector.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/context_injector.py), providing chunk formatting, source markers (`[1] source#chunk_index`), dynamic token budget calculation, and grounded prompt augmentation.
- **Key Modules & Functions**:
  1. `format_chunk(index, chunk, style)`: Formats individual retrieved chunks with unambiguous indexed citation markers.
  2. `assemble_context(chunks, max_tokens)`: Iterates through ranked chunks, measures exact BPE tokens via `tiktoken`, and terminates when reaching the context budget ceiling.
  3. `build_prompt(question, retrieved_chunks, max_context_tokens)`: Synthesizes a grounded prompt with strict anti-hallucination instructions and explicit fallback directives.
  4. `ContextInjector`: Object-oriented engine managing model context windows (e.g., 8,000 tokens) with dedicated allocations for context (5,000), generation output (1,500), and instructions (800).
- **Interactive Demonstration**: Run `python context_injection_demo.py` to:
  1. Demonstrate standard, detailed, and compact chunk formatting styles.
  2. Simulate token budget overflow scenarios and inspect dropped chunk audit details.
  3. Render complete grounded prompts and calculate token telemetry metrics.
  4. Export sample augmented prompts to [`sample_augmented_prompt.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_augmented_prompt.json) and [`sample_augmented_prompt.txt`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_augmented_prompt.txt).
- **Automated Test Suite**: Run `python test_context_injector.py` to verify 12 unit and integration test cases covering formatting, token budgets, and prompt structures.
- **Comprehensive Technical Report**: Full architectural flowcharts, token budget mathematical models, overflow mitigation strategies, and video script outline available in [`CONTEXT_INJECTION_PROMPT_AUGMENTATION_REPORT.md`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/CONTEXT_INJECTION_PROMPT_AUGMENTATION_REPORT.md).

## Grounded Answer Generation (Assignment 3.39)

- **Grounded Generation & Verification Engine**: Implemented in [`src/grounded_generator.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/grounded_generator.py), synthesizing factual responses strictly derived from retrieved evidence with deterministic citations, automated claim-to-chunk verification, and missing-context fallback guards.
- **Key Capabilities & Functions**:
  1. `generate_grounded_answer(question, retrieved_chunks)`: Generates answers using only injected context with inline citations (`[1]`, `[2]`).
  2. `verify_grounding(answer, retrieved_chunks)`: Analyzes word overlap and citation validity, computing a quantitative `grounding_score` [0.0 - 1.0] and detecting unsupported claims.
  3. `answer_query(question, retriever, k)`: Detects empty or insufficient retrieval states and safely returns: *"I don't have enough information in the provided context."*
  4. `generate_ungrounded_answer(question)`: Executes baseline zero-shot LLM queries without context to demonstrate ungrounded parametric completion behavior.
  5. `compare_grounded_vs_ungrounded(question, retrieved_chunks)`: Performs empirical side-by-side benchmarking of with-retrieval vs without-retrieval outputs.
  6. `GroundedGenerator`: Object-oriented orchestrator for retrieval, generation, verification, and comparison.
- **Interactive Demonstration**: Run `python grounded_generation_demo.py` to:
  1. Generate grounded answers from injected context and run automated accuracy audits.
  2. Test out-of-domain queries to verify missing-context fallback behavior.
  3. Display side-by-side comparison tables across multiple domain queries.
  4. Export complete structured evaluation payload to [`sample_grounded_answers.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_grounded_answers.json).
- **Automated Test Suite**: Run `python test_grounded_generator.py` to verify 10 unit and integration test cases.
- **Comprehensive Technical Report**: Complete architectural diagrams, empirical comparison tables, verification methods, and video walkthrough script available in [`GROUNDED_ANSWER_GENERATION_REPORT.md`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/GROUNDED_ANSWER_GENERATION_REPORT.md).

## Source Citation & Attribution (Assignment 3.40)

- **Source Citation & Traceability Engine**: Implemented in [`src/source_tracer.py`](file:///d:/Project/Knovera/src/source_tracer.py), pairing generated factual claims with inline citation markers (`[1]`, `[2]`), mapping markers to document metadata, verifying exact character offsets and line spans against raw source text, and auditing for fabricated markers.
- **Interactive Demonstration**: Run `py source_citation_demo.py` to execute citation mapping, reverse offset verification, anti-fabrication audits, and export [`sample_cited_answers.json`](file:///d:/Project/Knovera/sample_cited_answers.json).
- **Automated Test Suite**: Run `py -m unittest test_source_citation.py` to verify 12 unit test cases.
- **Comprehensive Technical Report**: Full architectural diagrams, reverse-lookup offset algorithms, anti-fabrication guards, and video walkthrough script available in [`SOURCE_CITATION_ATTRIBUTION_REPORT.md`](file:///d:/Project/Knovera/SOURCE_CITATION_ATTRIBUTION_REPORT.md).

## Hallucination Guardrails & Refusal Handling (Assignment 3.41)

- **Hallucination Guardrail Engine**: Implemented in [`src/hallucination_guardrail.py`](file:///d:/Project/Knovera/src/hallucination_guardrail.py), preventing ungrounded generation by auditing retrieved context strength before LLM invocation.
- **Key Modules & Capabilities**:
  1. `retrieval_is_strong(chunks, min_top_score=0.72, min_supporting_chunks=1)`: Evaluates context relevance against similarity and support thresholds.
  2. `evaluate_retrieval_quality(chunks)`: Performs multi-signal diagnostic evaluation returning detailed metrics, refusal reasons (`TOP_SCORE_BELOW_THRESHOLD`, `NO_CHUNKS_RETRIEVED`, `INSUFFICIENT_SUPPORT`), and status codes (`refused_low_similarity`, `refused_empty_context`).
  3. `guarded_answer(question, chunks)`: Intercepts weak retrieval states to return standard non-hallucinatory refusals (*"I don't have enough reliable context to answer that."*) while allowing strong context queries to pass to grounded generation with citations.
  4. `HallucinationGuardrail`: Object-oriented guardrail manager maintaining thresholds and logging telemetry (evaluated, refused, answered counts, refusal rates).
  5. `tune_guardrail_thresholds(eval_dataset)`: Dynamic threshold tuning utility optimizing F1-scores for answerable vs unanswerable query classification.
  6. `RAGPipeline.guarded_query(query)`: Integrated guardrail execution in end-to-end RAG pipeline workflows.
- **Interactive Demonstration**: Run `py hallucination_guardrail_demo.py` to inspect retrieval diagnostics, refusal cases, confident answered cases, guardrail telemetry, and export [`sample_guardrail_outputs.json`](file:///d:/Project/Knovera/sample_guardrail_outputs.json).
- **Automated Test Suite**: Run `py -m unittest test_hallucination_guardrail.py` to execute 16 unit tests covering empty, weak, low-similarity, custom threshold, and pipeline integration cases.
- **Comprehensive Technical Report**: Full pre-generation workflow diagrams, threshold tuning analysis, refusing vs answering trade-off matrices, high-stakes domain rationale, and benchmark results available in [`HALLUCINATION_GUARDRAILS_REFUSAL_REPORT.md`](file:///d:/Project/Knovera/HALLUCINATION_GUARDRAILS_REFUSAL_REPORT.md).

## Conversational RAG & Follow-Up Context (Assignment 3.42)

- **Conversational RAG Engine**: Implemented in [`src/history_manager.py`](file:///d:/Project/Knovera/src/history_manager.py) and [`src/rag_pipeline.py`](file:///d:/Project/Knovera/src/rag_pipeline.py), handling multi-turn dialogue history, standalone query rewriting for ambiguous follow-up questions, and token budgeting.
- **Interactive Demonstration**: Run `py conversational_rag_demo.py` to compare naive follow-up retrieval against standalone rewritten query retrieval, inspect dialogue turn histories, and export [`sample_conversational_rag.json`](file:///d:/Project/Knovera/sample_conversational_rag.json).
- **Automated Test Suite**: Run `py -m unittest test_conversational_rag.py` to execute unit and integration tests.
- **Comprehensive Technical Report**: Full multi-turn architecture diagrams, empirical retrieval score lift comparisons, and long-conversation grounding strategies available in [`CONVERSATIONAL_RAG_FOLLOWUP_CONTEXT_REPORT.md`](file:///d:/Project/Knovera/CONVERSATIONAL_RAG_FOLLOWUP_CONTEXT_REPORT.md).

## RAG Evaluation & Answer Quality Scoring (Assignment 3.43)

- **RAG Answer Quality Evaluation Engine**: Implemented in [`src/rag_evaluator.py`](file:///d:/Project/Knovera/src/rag_evaluator.py), providing end-to-end evaluation of the RAG pipeline across three key quality dimensions:
  1. `judge_expected_points(answer, expected_points)`: Quantifies **Correctness** by matching expected key factual points against the synthesized answer.
  2. `judge_grounding(answer, context)`: Quantifies **Grounding** by evaluating whether answer claims are supported by context (or safely refused when context is absent).
  3. `check_citations(citations, expected_sources)`: Quantifies **Citation Accuracy** ($F_1$ score) by validating inline cited sources against expected ground-truth sources.
  4. `diagnose_failure_cause(correctness, grounding, citation_accuracy)`: Diagnoses root causes of quality degradation (`WEAK_RETRIEVAL`, `UNSUPPORTED_DETAILS`, `BAD_CITATIONS`, `MISSING_EXPECTED_ANSWER_POINTS`).
  5. `summarize_evaluations(rows)`: Summarizes overall quality scores, averages, and isolates failure cases.
- **Structured Test Set**: [`data/sample_rag_test_set.json`](file:///d:/Project/Knovera/data/sample_rag_test_set.json) containing questions, expected points, and expected sources across academic policies, media rules, vector specs, chunking standards, and out-of-scope refusals.
- **Interactive Demonstration**: Run `py rag_evaluation_demo.py` to evaluate the RAG system, print the detailed per-query quality table, compute aggregate metric summaries, and export [`sample_rag_evaluation_results.json`](file:///d:/Project/Knovera/sample_rag_evaluation_results.json) and [`outputs/rag_evaluation_output.txt`](file:///d:/Project/Knovera/outputs/rag_evaluation_output.txt).
- **Automated Test Suite**: Run `py -m unittest test_rag_evaluator.py` to verify metric scoring, refusal grounding, citation accuracy, failure diagnosis, and summary calculations.
- **Comprehensive Technical Report**: Complete evaluation workflow diagrams, metric definitions, empirical score tables, failure case root-cause analysis, weakest dimension improvement blueprint, and 3–5 min video walkthrough script available in [`RAG_EVALUATION_ANSWER_QUALITY_REPORT.md`](file:///d:/Project/Knovera/RAG_EVALUATION_ANSWER_QUALITY_REPORT.md).

## Backend API for the RAG Service (Assignment 3.44)

- **FastAPI Backend REST Service**: Implemented in [`src/rag_api.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/rag_api.py) and [`src/api_config.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/api_config.py), establishing a stable, production-grade REST contract between the Knovera RAG pipeline and frontend/external client consumers.
- **Key Endpoints & Features**:
  1. `POST /query`: Primary question answering endpoint accepting `QueryRequest` and returning structured `QueryResponse` with grounded answer, source citations (`source`, `chunk_id`, `score`, `rank`), execution status (`answered`, `refused_empty_context`, `refused_low_similarity`), and per-stage latency breakdown.
  2. `GET /health`: Service health and readiness endpoint monitoring active ChromaDB collection status, embedding models, and chat LLMs.
  3. `GET /config`: Sanitized runtime configuration endpoint with credential masking.
  4. `GET /`: Service overview and OpenAPI interactive documentation routing (`/docs` and `/redoc`).
  5. **Strict Input Validation & HTTP Status Codes**: Validates query lengths (3 to 1000 characters), rejects empty queries, and returns standard HTTP codes (`200 OK`, `400 Bad Request`, `422 Unprocessable Entity`, `500 Internal Server Error`).
  6. **Environment-Driven Configuration**: Dynamically loads API keys, embedding models, vector store directories, and port bindings via `APIConfig` and `.env`.
- **Interactive Demonstration**: Run `python rag_api_demo.py` to execute health checks, configuration inspection, grounded queries, guardrail refusals, input validation tests, and export [`sample_api_query_response.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_api_query_response.json) and [`outputs/api_query_sample.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/outputs/api_query_sample.json).
- **Automated Test Suite**: Run `python test_rag_api.py` (or `python -m unittest test_rag_api.py`) to verify all 14 unit and integration test cases covering request schemas, validation limits, error handlers, and environment loading.
- **Comprehensive Technical Report**: Complete REST endpoint specifications, Pydantic schemas, validation error tables, environment configuration guide, frontend integration TypeScript/React snippets, and video walkthrough script available in [`BACKEND_API_RAG_SERVICE_REPORT.md`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/BACKEND_API_RAG_SERVICE_REPORT.md).

## Document Upload & Indexing Endpoint (Assignment 3.45)

- **Document Upload & Ingestion Engine**: Implemented in [`src/document_indexer.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/document_indexer.py) and [`src/rag_api.py`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/src/rag_api.py), enabling runtime expansion of the vector knowledge base through a unified REST endpoint without service restarts.
- **Key Capabilities & Architecture**:
  1. `POST /documents` (and alias `POST /upload`): Accepts `multipart/form-data` uploads (`.txt`, `.md`, `.pdf`, `.html`), validates file types and sizes, stores documents securely to `uploads/`, and coordinates multi-stage ingestion.
  2. **End-to-End Processing Pipeline**: Automates loading (`DocumentLoader`), cleaning (`TextCleaner`), token-aware chunking (`TokenChunker`), metadata tagging (`ChunkTagger`), batch embedding (`EmbeddingGenerator`), and atomic vector database upsert (`VectorDatabase`).
  3. **Immediate Runtime Searchability**: Newly indexed documents are immediately searchable via `POST /query` without server restart or index rebuilds.
  4. **Strict Error Handling & Validation**: Returns `HTTP 415` for unsupported formats, `HTTP 400` for empty/corrupt files, `HTTP 413` for files exceeding size limits (default 10 MB), and `HTTP 500` for unexpected pipeline failures.
- **Interactive Demonstration**: Run `python document_upload_demo.py` to observe pre-upload query state, upload a new policy document, confirm immediate post-upload query retrieval with source attribution, test validation error cases, and export [`sample_document_upload_run.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/sample_document_upload_run.json) and [`outputs/sample_upload_query_result.json`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/outputs/sample_upload_query_result.json).
- **Automated Test Suite**: Run `python test_document_upload.py` to verify 10 unit and integration test cases covering format validation, empty file rejections, oversized payload guards, and runtime searchability.
- **Comprehensive Technical Report**: Full architecture flowcharts, API schemas, large document asynchronous queue strategies, and video walkthrough script available in [`DOCUMENT_UPLOAD_INDEXING_REPORT.md`](file:///c:/Users/K%20Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/DOCUMENT_UPLOAD_INDEXING_REPORT.md).

## Clean-Run Proof

The setup instructions have been tested on a fresh setup and everything runs cleanly without any conflicts or missing dependencies.

