# Code Mesh Research Note: The Deterministic Context Layer

> **Note from the author:** This directory is a research artifact from the LlamaPReview context work. While the traditional AI code review SaaS model as it stands has seemingly outlived its time in the era of "Vibe Coding," the deterministic-context thesis developed here remains a powerful foundation. This is not an active LlamaPReview product roadmap, but if you've resonated with this approach, I'd be incredibly honored if you took a moment to look at [DocMason](https://github.com/JetXu-LLM/DocMason), where our passion for building developer tools continues.

> **"Stop searching. Start traversing."**

## 1. The Paradigm Shift

In our research (see `search_rag` and `agentic_rag`), we identified a fundamental flaw in applying standard RAG to software engineering: **Probabilistic Retrieval is insufficient for Code.**

*   **Search RAG** is like asking a librarian: *"Find books that feel similar to this one."* (Good for chat, bad for compilation).
*   **Code Mesh** is like using a GPS: *"Trace the exact path from Point A to Point B."*

**Code Mesh** is the architectural idea we explored for treating a repository not as a bag of text files, but as a **Semantic Network**.

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

## 4. Research Status

This directory represents the architectural definition of the Code Mesh strategy.

This public research repository currently preserves:
*   Architecture notes for deterministic traversal behavior.
*   The surrounding research context for why deterministic retrieval matters.

This repository does **not** include a production Code Mesh engine.

> **Note:** The broader context-intelligence direction from this research now has a different public focus. DocMason is the current active open-source focus from the same author, while this directory remains here for transparency and reference.

---

*This specification defines the "Strategy C" in our research on Context Intelligence.*
