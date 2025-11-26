# LlamaPReview Context Intelligence Research

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)

## 📖 Introduction

**The "Context Instability" Problem**

The primary bottleneck in AI-assisted software engineering isn't the intelligence of the LLM—it's the **instability of the context retrieval**.

When we ask an AI to review a PR, we are essentially asking it to find a needle in a haystack. Current industry standards rely on **Probabilistic Retrieval** (Vector Search / Keyword Search). This works for chat, but fails for code engineering where strict dependency logic is required.

This repository documents our research journey through **Strategy A** and **Strategy B**, and explains why we ultimately moved towards the **Code Mesh Architecture**.

---

## 🧪 Running the Experiments

This repository includes a benchmark suite to compare the three context retrieval strategies.

1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    cp .env.example .env  # Add your GITHUB_TOKEN and DEEPSEEK_API_KEY
    ```

2.  **Run Single Strategy (CLI):**
    ```bash
    python main.py https://github.com/psf/requests/pull/6666 --strategy agent
    ```

3.  **Run Comparison Benchmark:**
    Generate a side-by-side report of Search vs. Agent vs. Mesh:
    ```bash
    python experiments/run_comparison.py https://github.com/psf/requests/pull/6666
    ```
    *Output: `comparison_report_6666.md`*

---

## 🏗️ The Probabilistic Approaches (This Repo)

This codebase contains the implementations of two common RAG patterns we evaluated:

### Strategy A: Search-based RAG (The Baseline)
*Located in `strategies/search_rag/`*
*   **Mechanism:** Regex + GitHub Search API.
*   **Verdict:** Fast but low recall. It misses implicit dependencies (e.g., aliased imports or dynamic dispatch).

### Strategy B: Agentic RAG (The Explorer)
*Located in `strategies/agentic_rag/`*
*   **Mechanism:** ReAct Agents exploring the file tree.
*   **Verdict:** High precision but prohibitive cost/latency. O(N) complexity makes it unscalable for large monoliths.

> **Note:** These implementations are provided as research artifacts to demonstrate the limitations of non-deterministic retrieval.

---

## 🔮 The Future: The Code Mesh Paradigm

Our research concluded that **you cannot solve a structural problem with a probabilistic tool.**

To achieve 100% context consistency, we are shifting our focus to **Strategy C: The Code Mesh**.

### What is Code Mesh?
Code Mesh is not just a tool; it is a **deterministic infrastructure layer** for AI coding agents.

*   **From Text to Graph:** Instead of treating code as flat text files, Code Mesh parses the repository into a semantic graph (Nodes: Definitions, Edges: References/Calls).
*   **Deterministic Navigation:** It replaces "Searching" with "Traversing". When an LLM needs to know "Who calls this function?", it doesn't guess—it simply follows the edge.
*   **O(1) Efficiency:** Context retrieval becomes a direct lookup operation, independent of the repository size.

*The architectural specification and implementation of Code Mesh will be detailed in our upcoming technical series.*

---

## 👤 Author

**Jet Xu**
*Architect of LlamaPReview & Code Mesh*

This research is part of the broader mission to build the **Deterministic Context Layer** for AI.

---

*Disclaimer: This code is provided for research and educational purposes.*
