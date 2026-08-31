# Text Extraction & Cleaning Pipeline Notes (Assignment 3.20)

## 1. Why Raw Extracted Text Needs Cleaning Before Use

In a Retrieval-Augmented Generation (RAG) system, **"Garbage In, Garbage Out"** is brutally literal. Document loaders extract raw text from diverse formats (PDF, HTML, TXT, Markdown), but the resulting strings are laden with extraction noise:
- **Repeated Page Headers & Footers**: "Page 3 of 12", "KNOVERA CONFIDENTIAL", copyright notices repeated across dozens of pages.
- **Broken Mid-Word Line Wraps**: Hyphenated words split across page margins (e.g., `transfor-\nmation`, `perfor-\nmance`).
- **Encoding Glitches & Mojibake**: Garbled characters resulting from UTF-8/ISO-8859-1 decoding mismatches (e.g., `â€™` for `'`, `â€œ` for `"`, `â€”` for `—`).
- **Runaway Whitespace**: Irregular tab stops, dozens of consecutive spaces, and excessive blank lines from PDF layout extraction.

### Impact on Downstream RAG Components:
1. **Embedding Model Pollution**: Dense vector models (e.g., `text-embedding-3-small`) generate high-dimensional vectors based on every token. When boilerplate headers appear across 50 distinct document chunks, vector search retrieves chunks due to shared boilerplate rather than relevant domain answers.
2. **Split Tokenization & Semantic Loss**: Broken line wraps like `perfor-\nmance` get tokenized into disjoint subwords `['per', 'for', '-', 'mance']`, completely destroying the semantic vector for the word "performance".
3. **Context Window Waste**: Ingesting repetitive navigation text and blank lines inflates token counts, consuming LLM context window space that should be reserved for factual domain knowledge.

---

## 2. Knovera Text Cleaning Pipeline Architecture

Knovera's `TextCleaner` (`src/text_cleaner.py`) applies a standardized, deterministic 5-stage transformation pipeline:

```
+------------------+     +-----------------------+     +------------------------+
|  Raw Extracted   | --> | 1. Line Endings       | --> | 2. Unicode (NFKC) &    |
|  Document Text   |     |    (\r\n, \r -> \n)   |     |    Mojibake Repair     |
+------------------+     +-----------------------+     +------------------------+
                                                                   |
                                                                   v
+------------------+     +-----------------------+     +------------------------+
| Uniform Cleaned  | <-- | 5. Whitespace &       | <-- | 3. De-hyphenate Broken |
| Document Output  |     |    Blank Line Normal. |     |    Line-Wraps          |
+------------------+     +-----------------------+     +------------------------+
                                     ^                             |
                                     |                             v
                                     +-----------------+ 4. Strip Boilerplate   |
                                                       |    (Headers, Footers)  |
                                                       +------------------------+
```

### Stage 1: Line Ending Normalization
- Standardizes all Windows CRLF (`\r\n`) and legacy Mac CR (`\r`) line endings to standard Unix LF (`\n`).
- Prevents cross-platform newline discrepancies during downstream regular expression parsing and chunking.

### Stage 2: Unicode NFKC & Mojibake Resolution
- **Unicode NFKC Normalization**: `unicodedata.normalize("NFKC", text)` converts compatibility characters and composite forms into their standard canonical equivalents.
- **Mojibake Replacement Dictionary**: Detects and replaces multi-byte UTF-8 decoding corruptions (e.g., `â€™` $\rightarrow$ `'`, `â€œ` $\rightarrow$ `"`, `â€”` $\rightarrow$ `—`, `\xa0` $\rightarrow$ space, `\ufeff` $\rightarrow$ stripped).

### Stage 3: Broken Line-Wrap De-Hyphenation
- Employs regex `r"([a-zA-Z]{2,})-\n([a-zA-Z]{2,})"` $\rightarrow$ `r"\1\2"` to reconnect words split across line breaks during PDF extraction (e.g., `integra-\ntion` becomes `integration`).
- **Semantic Safeguard**: Retains intentional inline compound hyphens on the same line (e.g., `state-of-the-art`, `fine-tuned`) without modification.

### Stage 4: Boilerplate & Page Header/Footer Stripping
- Removes repetitive page counters: `Page \d+ of \d+`, `[Page \d+]`, `- \d+ -`.
- Strips repeated confidential markings: `KNOVERA CONFIDENTIAL - DO NOT DISTRIBUTE`.
- Eliminates navigation breadcrumbs and UI artifacts: `Home > Docs > ...`, `[Back to Top]`.

### Stage 5: Whitespace Normalization
- Converts non-standard tabs to single spaces.
- Collapses consecutive internal spaces while preserving necessary leading indentation for code blocks and bulleted lists.
- Strips trailing line whitespace and collapses 3+ consecutive newlines into clean 2-newline paragraph breaks (`\n\n`).

---

## 3. How Poor Cleaning Degrades Retrieval Quality

| Pipeline Defect | Raw Text Example | Vector Retrieval Consequence |
|---|---|---|
| **Repeated Boilerplate** | `"KNOVERA CONFIDENTIAL - Page 3 of 12"` on every page | Chunks containing boilerplate match queries containing generic corporate terms, pushing true answer chunks out of Top-$K$. |
| **Broken Line Wraps** | `"down-\nstream retri-\neval"` | Embedding tokenizer fails to recognize `"downstream"` and `"retrieval"`, drastically lowering cosine similarity with user queries. |
| **Encoding Mojibake** | `"Weâ€™ve observed 95% accuracy"` | Query for `"We've observed"` fails exact or semantic match due to multi-byte noise characters. |
| **Runaway Whitespace** | 10 blank lines between sections | Chunker splits paragraphs arbitrarily into empty or low-information chunks. |

---

## 4. The Principle of Corpus-Wide Cleaning Consistency

Applying inconsistent cleaning across a document corpus introduces **asymmetric retrieval bias**:
- If Document A is cleaned of boilerplate but Document B retains boilerplate, Document B may receive artificial score boosts on certain queries while suffering severe penalties on domain-specific queries.
- Running `TextCleaner.clean_corpus()` ensures that every ingested document — regardless of whether it originated as a PDF, Markdown, HTML, or TXT file — undergoes the exact same transformations, producing uniform vector embeddings.

---

## 5. Avoiding Over-Cleaning: Edge Cases & Semantic Preservation

Cleaning must be balanced; over-aggressive cleaning destroys critical semantic information:
1. **Preserving Punctuation**: Punctuation marks (`.`, `?`, `!`, `,`, `:`, `;`) define grammatical structure and semantic boundaries. Stripping all punctuation turns sentences into unreadable bags-of-words.
2. **Preserving Numbers & Units**: Numeric values (`94.8%`, `$15,000`, `v2.4.0`) are vital facts in technical and financial documents.
3. **Preserving Code Blocks & Indentation**: Code snippets (e.g. JSON payloads, Python snippets) require preserved braces `{}`, quotes `""`, and indentation to retain validity.
4. **De-hyphenation Edge Cases**: Distinguishing between hyphenated split words at line breaks (`concep-\ntual`) and intentional hyphenated compound words (`state-of-the-art`). Knovera's cleaner only removes hyphens when followed immediately by a newline and subsequent word letters.

---

## 6. Corpus Before vs. After Cleaning Metrics

| Document | Format | Raw Chars | Clean Chars | Chars Removed | Reduction % | Primary Noise Removed |
|---|---|---|---|---|---|---|
| `noisy_document_sample.txt` | `.txt` | 1,694 | 1,389 | 305 | 18.00% | Repeated headers, "Page X of Y", mojibake `â€™`, split line wraps, nav breadcrumbs |
| `api_documentation.md` | `.md` | 945 | 918 | 27 | 2.86% | Trailing whitespace, runaway newlines |
| `customer_policy.txt` | `.txt` | 992 | 992 | 0 | 0.00% | Clean document preserved without distortion |
| `employee_handbook.pdf` | `.pdf` | 558 | 558 | 0 | 0.00% | Preserved formatting, verified clean |
| `release_notes.html` | `.html` | 493 | 493 | 0 | 0.00% | Preserved list items and structure |

---

## 7. Video Walkthrough Outline (3–5 Minutes)

When recording your video submission, follow this structured outline:

1. **Introduction & Problem Statement (0:00 - 0:45)**:
   - State your name and project: Knovera RAG Assistant — Text Extraction & Cleaning Pipeline.
   - Explain why raw extracted text cannot be embedded as-is: repeated headers, "Page X of Y" footers, mojibake, and split hyphenated words pollute vector embeddings ("Garbage in, garbage out").
2. **Cleaning Pipeline Architecture (0:45 - 1:45)**:
   - Open `src/text_cleaner.py`.
   - Walk through the 5 pipeline stages:
     - `normalize_line_endings()`: Windows/Mac to Unix `\n`.
     - `fix_encoding_artifacts()`: Unicode NFKC and mojibake dictionary replacement.
     - `repair_broken_line_wraps()`: Regex de-hyphenation of words split across lines.
     - `strip_boilerplate()`: Regex purging of repeated headers, footers, page numbers, and navigation.
     - `collapse_whitespace()`: Normalizing spaces/tabs and collapsing runaway newlines.
3. **Execution Demo & Before/After Evidence (1:45 - 2:45)**:
   - Run `python text_cleaning_demo.py` in the terminal.
   - Show the step-by-step transformation on `sample_raw`.
   - Point out the corpus before/after metrics table and explain character reductions.
   - Show how `noisy_document_sample.txt` had 305 characters of noise stripped (18% reduction).
4. **Preventing Over-Cleaning & Edge Cases (2:45 - 3:30)**:
   - Explain how Knovera avoids over-cleaning: code blocks (JSON syntax), punctuation, and intentional hyphenated terms (`state-of-the-art`) are preserved intact.
   - Mention how consistent corpus cleaning ensures fair and accurate vector retrieval.
5. **Conclusion & GitHub PR (3:30 - 4:00)**:
   - Confirm pipeline integration with `DocumentLoader`.
   - Show the output report `outputs/text_cleaning_output.txt`.
   - Close with confirmation of the public GitHub PR.
