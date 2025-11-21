# LlamaPReview Context Intelligence Research

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)

## 📖 Introduction

**Why are AI code reviewers inconsistent?**

The primary bottleneck in AI-assisted software engineering isn't the intelligence of the LLM—it's the **instability of the context**.
- When a Pull Request is isolated (intra-file changes), AI tools achieve >80% accuracy.
- When a Pull Request involves cross-file dependencies (inter-file impacts), accuracy drops significantly due to missing context.

This repository contains the research artifacts and experimental implementations behind **LlamaPReview's** journey to solve this problem. It documents our exploration of two distinct architectural approaches to Context RAG (Retrieval-Augmented Generation):

1.  **Search-based RAG:** The industry standard "needle-in-a-haystack" approach.
2.  **Agentic RAG:** A "brute-force exploration" approach using ReAct agents.

> **Note:** This repository serves as a companion to our upcoming technical deep-dive series. The code here is structured as research prototypes to demonstrate architectural patterns.

---

## 🏗️ Architectures Explored

### Strategy A: Search-based RAG (The Baseline)
*Located in `strategies/search_rag/`*

This approach mimics how most current AI developer tools work. It relies on generating keywords from the PR diff and querying the GitHub Search API.

*   **Mechanism:** LLM Query Generation -> GitHub Code Search API -> Regex Filtering -> Context Window.
*   **Key Components:**
    *   `QueryGenerator`: Uses heuristics to generate high-recall search terms.
    *   `CodeContextExtractor`: Language-agnostic regex patterns to extract function/class definitions.
*   **The Trade-off:** Fast and cheap, but suffers from **Low Recall**. It often misses semantic relationships (e.g., aliased imports) and struggles with "Zero-Result" queries.

### Strategy B: Agentic RAG (The Explorer)
*Located in `strategies/agentic_rag/`*

This approach utilizes a **ReAct (Reason-Act)** loop powered by reasoning models (e.g., DeepSeek-Reasoner). The agent mimics a human reviewer, navigating the file tree step-by-step.

*   **Mechanism:** Think (Plan) -> Act (Read File/List Dir) -> Observe (Evaluate Relevance) -> Reflect -> Decide.
*   **Key Components:**
    *   `PRContextCollector`: The core engine managing the ReAct loop.
    *   `QualityEvaluator`: An automated scoring system to judge context completeness and sufficiency.
*   **The Trade-off:** High precision and reasoning depth, but suffers from **High Latency and Cost**. A single review involves multiple LLM round-trips, making it O(N) in complexity.

---

## 📂 Project Structure

```text
llamapreview-context-research/
├── core/                    # Shared utilities (GitHub client, PR processing)
├── strategies/
│   ├── search_rag/          # Implementation of Search-based retrieval
│   └── agentic_rag/         # Implementation of Agent-based retrieval
├── experiments/             # Comparison scripts and notebooks
└── main.py                  # CLI entry point
```

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   GitHub Access Token (Fine-grained token with repo read permissions)
*   DeepSeek API Key (or compatible OpenAI-format endpoint)

### Installation

```bash
git clone https://github.com/YourUsername/llamapreview-context-research.git
cd llamapreview-context-research
pip install -r requirements.txt
```

### Usage (CLI)

You can run the context retrieval pipeline using either strategy:

```bash
# Run Search-based Strategy
python main.py --repo owner/repo --pr 123 --strategy search

# Run Agentic Strategy
python main.py --repo owner/repo --pr 123 --strategy agent
```

---

## 🔮 The "Graph" Horizon

While both Search and Agentic strategies offer solutions, our research indicates that **Context Instability** persists due to the probabilistic nature of these methods.

At **LlamaPReview**, we are currently moving towards **Strategy C: Repository Graph RAG**. By statically analyzing the repository to build a deterministic dependency graph (Nodes: Functions, Edges: Calls), we aim to achieve O(1) lookup efficiency with 100% consistency.

*More details on the Graph approach will be discussed in future publications.*

---

## 👤 Author

**Jet Xu**
*Creator of LlamaPReview*

This research is part of the broader mission to build the next generation of context-aware AI developer tools.

---

*Disclaimer: This code is provided for research and educational purposes. It represents experimental snapshots and may differ from the production codebase of LlamaPReview.*
