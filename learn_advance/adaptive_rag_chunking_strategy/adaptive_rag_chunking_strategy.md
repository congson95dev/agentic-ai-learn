# Adaptive RAG --- Chunking Strategy Selector

## 1. Goal

Implement a small **Adaptive Chunking Strategy Selector** in the
existing project.

This is a learning/prototype task. **Do not implement or execute a real
RAG pipeline.** The deliverable may read and extract uploaded documents,
but it only recommends a strategy; it does not perform chunking.

The feature should accept a document (or document metadata/content),
analyze it, select the most appropriate chunking strategy, and output
only:

-   Selected strategy
-   Short reason for the selection
-   Optional confidence score
-   Basic analysis signals used for the decision

It must **not** split the document into chunks, generate embeddings,
call a vector database, retrieve documents, or call an LLM to answer
questions.

The implementation also provides a small FastAPI upload UI/API. A user
uploads a document, the backend extracts text and parser metadata, and the
selector returns one strategy name, a short reason, confidence, and basic
signals. LLM-based metadata analysis is optional future work and is not a
dependency of this prototype.

## 2. Strategies to support

Support these five strategy labels:

1.  `fixed_size`
    -   Split by a fixed character/token budget.
    -   Use as a simple fallback when the document is plain text and has
        little useful structure.
2.  `recursive`
    -   Split recursively using separators such as paragraph, newline,
        whitespace, and finally character boundaries.
    -   Use as the general-purpose default for ordinary prose and
        unstructured text.
3.  `semantic`
    -   Split based on semantic similarity between neighboring sentences
        or passages.
    -   Consider when the content contains multiple distinct topics or
        abrupt topic transitions that are not reliably represented by
        document structure.
    -   This prototype only selects the strategy; do not calculate
        embeddings or semantic similarity.
4.  `structure_based`
    -   Split according to the document's native structure, such as
        Markdown headings, HTML elements, PDF headings/sections, or
        source-code functions/classes.
    -   Prefer this when reliable structural boundaries are detected.
5.  `agentic`
    -   Use an LLM/agent to reason about complex document segmentation.
    -   Consider only for unusually complex documents where simple
        structural or heuristic selection is insufficient.
    -   For this prototype, do not call an LLM. Select this strategy
        only through explicit, explainable rules/heuristics, and make
        clear that it is a recommendation rather than an actual agentic
        segmentation.

## 3. Input and output

First inspect the existing repository and follow its current language,
framework, folder structure, naming conventions, and coding style. Reuse
existing document-loading or parsing utilities if they are already
present, but do not add unnecessary dependencies.

Design a small selector interface. Adapt the exact interface to the
existing codebase, but it should conceptually accept:

-   `file_name` or `file_type`
-   `content` (optional but useful)
-   `metadata` (optional; e.g. parser-detected headings, page count,
    source type)

Example input:

``` python
{
    "file_name": "architecture.md",
    "file_type": "markdown",
    "content": "# Overview\n...\n## Architecture\n...",
    "metadata": {}
}
```

Return a structured result similar to:

``` json
{
  "strategy": "structure_based",
  "reason": "The document contains clear Markdown heading boundaries.",
  "confidence": 0.92,
  "signals": {
    "file_type": "markdown",
    "has_headings": true,
    "has_code_blocks": false,
    "topic_complexity": "unknown"
  }
}
```

The exact confidence scale is `0.0` to `1.0`. Confidence is an estimate
from the selector's rules, not a machine-learned probability. If
confidence is not meaningful, omit it rather than inventing precision.

Also provide a simple way to run the selector locally (for example, a
CLI command or a small existing-app entry point) and print the result as
readable JSON. The implementation includes a minimal FastAPI upload UI and
JSON API because the intended workflow is document upload:

- `GET /` — upload form;
- `POST /analyze` — HTML result;
- `POST /api/analyze` — JSON result;
- `GET /health` — health check.

### 3.1 Real document input and extraction

The backend accepts real uploaded files, not only a caller-provided string:

- UTF-8 text/code/Markdown/HTML files through the standard library;
- PDF through `pypdf`;
- DOCX through `docx2txt`.

Extraction is isolated from strategy selection. The reader returns:

```python
content, parser_metadata = read_document(path)
```

The selector then receives `content` and `parser_metadata`. A PDF without a
text layer (for example, a scanned image) is reported as having no
extractable text rather than silently treated as reliable structure.

## 4. Selection logic

Implement deterministic, explainable rules. Prefer the following order:

### A. Structure-based detection

Choose `structure_based` when the document has reliable native
structure, for example:

-   Markdown with multiple meaningful headings (`#`, `##`, etc.).
-   HTML with meaningful sectioning elements or a clear DOM hierarchy.
-   Source code with identifiable functions/classes/modules.
-   Parsed PDF/DOCX content where headings or section boundaries are
    explicitly available in metadata.

Do not select `structure_based` merely because the file extension is PDF
or DOCX. The structure must actually be detectable or supplied by the
parser.

### B. Agentic recommendation

Choose `agentic` only when there are explicit signs of unusually complex
content, such as:

-   Mixed content types and nested structures that cannot be handled
    reliably by the available structure-aware rules.
-   Dense cross-references or interdependent sections spanning the
    document.
-   Highly irregular layouts or fragmented content where deterministic
    boundaries are inadequate.

Be conservative. Do not use `agentic` as a synonym for "long document"
or "many topics." Since no LLM is called in this prototype, explain that
the selector recommends an agentic approach but does not perform it.

### C. Semantic detection

Choose `semantic` when there is evidence of multiple distinct topics or
substantial topic transitions, but no reliable native structure to
preserve.

For this prototype, use lightweight heuristics only (e.g. paragraph
boundaries, topic/section cues, or simple keyword distribution). Do not
add embedding models or external API calls. If topic complexity cannot
be assessed reliably, do not force this choice.

### D. Recursive default

Choose `recursive` for ordinary prose, reports, articles, and general
text when no stronger signal applies.

This should be the normal fallback for general-purpose RAG documents.

### E. Fixed-size fallback

Choose `fixed_size` only for very simple/plain text inputs where there
are no useful paragraph or structural boundaries, or where the input is
effectively a flat text stream.

Avoid choosing it over `recursive` for ordinary well-formed prose.

## 5. Important decision principles

-   The selector chooses **one recommended strategy per document**.
-   Do not run all strategies or generate chunks.
-   Do not claim that a strategy is objectively optimal; it is a
    recommendation based on observable signals.
-   Do not assume every PDF is structure-based or every long document
    needs agentic chunking.
-   Prefer deterministic rules over an LLM for this learning prototype.
-   Keep the implementation small and easy to understand. Avoid
    overengineering, agent frameworks, vector stores, embedding models,
    and unnecessary abstractions.
-   If input information is insufficient, choose `recursive` and explain
    that it is the safe general-purpose default.
-   Keep strategy selection separate from any existing RAG
    ingestion/retrieval code.
-   Prefer parser/rule-based metadata for clear signals. Do not require an
    LLM to produce `complex_layout` or `cross_references`; those fields may
    be supplied by an upstream parser while the selector remains
    deterministic.

## 6. Tests

Add unit tests for at least these cases:

  Input characteristics                              Expected strategy
  -------------------------------------------------- -------------------
  Markdown with multiple headings                    `structure_based`
  Source code with functions/classes                 `structure_based`
  Ordinary prose with paragraphs                     `recursive`
  Plain text with no meaningful boundaries           `fixed_size`
  Unstructured content with clear topic shifts       `semantic`
  Highly irregular, nested, interdependent content   `agentic`
  Unknown/insufficient input                         `recursive`

Tests should verify both the selected strategy and that the
reason/signals are present. Tests must not call external services or
require API keys.

## 7. Deliverables

1.  Implement the selector in the existing project.
2.  Add unit tests covering the cases above.
3.  Add a short README section explaining:
    -   What Adaptive Chunking Strategy Selection does.
    -   The five supported strategies.
    -   The selection rules and limitations.
    -   How to run the selector and see its output.
4.  Provide one or more sample inputs and their expected outputs.
5.  Provide a FastAPI upload UI and JSON endpoint for real files.
6.  Summarize changed files and the command used to run the tests.

## 8. Out of scope

Explicitly do not implement:

-   Actual chunk creation/splitting.
-   Embedding generation.
-   Vector database integration.
-   Retriever or similarity search.
-   LLM-based document analysis or agent execution.
-   Answer generation or a complete RAG workflow.
-   Performance benchmarking.
-   LLM-based document analysis is optional future work and is not required
    for this prototype.

**Acceptance criterion:** Given a sample document, running the selector
prints the recommended strategy and a short explanation. Nothing else in
the RAG pipeline needs to execute.
