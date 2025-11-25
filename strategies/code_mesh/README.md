# Code Mesh: The Deterministic Context Layer

> **"Stop searching. Start traversing."**

## 1. The Paradigm Shift

In our research (see `search_rag` and `agentic_rag`), we identified a fundamental flaw in applying standard RAG to software engineering: **Probabilistic Retrieval is insufficient for Code.**

*   **Search RAG** is like asking a librarian: *"Find books that feel similar to this one."* (Good for chat, bad for compilation).
*   **Code Mesh** is like using a GPS: *"Trace the exact path from Point A to Point B."*

**Code Mesh** is our proprietary infrastructure that treats a repository not as a bag of text files, but as a **Semantic Network**.

## 2. Architecture Specification

Unlike "Graph RAG" which typically deals with unstructured text entities, Code Mesh is built specifically for the rigid, logical structure of programming languages.

### The Data Model
The Mesh is a directed graph $G = (V, E)$ constructed via Static Analysis (AST Parsing):

*   **Nodes ($V$):** Semantic Symbols
    *   `DefinitionNode`: Classes, Functions, Structs.
    *   `FileNode`: Physical file boundaries.
*   **Edges ($E$):** Hard Dependencies
    *   `CALLS`: Function A calls Function B.
    *   `IMPORTS`: Module A imports Module B.
    *   `INHERITS`: Class A extends Class B.
    *   `INSTANTIATES`: Function A creates an instance of Class B.

### The Retrieval Mechanism: "Context Traversal"

Instead of "Similarity Search", Code Mesh performs **Graph Traversal** to build context.

**Scenario:** A user asks about `process_payment()`.

1.  **Anchor:** Locate `process_payment` node (Exact Match).
2.  **Upstream:** Traverse reverse edges to find *who calls* `process_payment` (Impact Analysis).
3.  **Downstream:** Traverse forward edges to find *what* `process_payment` relies on (Dependency Resolution).
4.  **Pruning:** Apply algorithms to remove standard library calls or irrelevant utilities.

## 3. Performance Comparison

| Feature | Search RAG (Vector) | Agentic RAG (ReAct) | Code Mesh (Graph) |
| :--- | :--- | :--- | :--- |
| **Recall** | ~70% (Misses implicit links) | ~95% | **100% (Deterministic)** |
| **Precision** | Low (High Noise) | Medium | **High (Zero Noise)** |
| **Latency** | Fast | Very Slow (Multi-step) | **Instant (O(1) Lookup)** |
| **Cost** | Low | High | **Low** |

## 4. Implementation Status

This directory represents the **architectural definition** of the Code Mesh strategy.

The core implementation includes:
*   **Polyglot Parser:** Based on Tree-sitter for multi-language support.
*   **Graph Builder:** NetworkX/Rust-based graph generation.
*   **Traversal Engine:** Algorithms for "Context Slicing".

> **Note:** The full implementation of the Code Mesh engine is the proprietary core of **[LlamaPReview](https://jetxu-llm.github.io/LlamaPReview-site/)**. It is currently deployed in production, powering thousands of accurate code reviews daily.

---

*This specification defines the "Strategy C" in our research on Context Intelligence.*