# 🛡️ ClauseGuard — Contract Clause Risk Analyzer

> **Clause-level AI risk detection for legal contracts.**
> Upload a PDF, DOCX, or TXT — get a clause-by-clause risk analysis with plain-English explanations in under a minute.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What It Does

Traditional contract-analysis tools classify the *whole document* as risky or safe. ClauseGuard analyzes **every clause individually**:

- **Segments** the contract into structured clauses with character-level offsets
- **Classifies** each clause into one of 100 legal categories (Termination, Confidentiality, Indemnification, IP Assignment, etc.)
- **Compares** each clause against a reference bank of 500 medoid exemplars using semantic similarity
- **Scores** risk using a 5-signal ensemble, including a hand-curated risky-language lexicon
- **Explains** every flagged clause in plain English via Gemini, with a concrete suggested action

The result is an interactive dashboard that highlights exactly which clauses need attention — and why.

---

## 🎯 Research Angle — Clause-Level, Not Document-Level

Most published work on legal NLP treats contracts as single classification units (document-level). **This project targets the gap between document-level and clause-level analysis:**

> Two Termination clauses may both be correctly classified as "Terminations" — but one says *"either party may terminate upon 30 days notice"* (standard), while the other says *"the Company may terminate immediately at its sole discretion without cause or notice"* (highly risky).
>
> **Classification alone cannot distinguish them. We need clause-level semantic + substantive analysis.**

ClauseGuard solves this with four complementary signals:

1. **Classifier confidence** — how sure the model is about clause type
2. **Semantic deviation** — how different the clause is from the standard reference
3. **Classifier-vs-reference mismatch** — a cross-signal disagreement indicator
4. **Substantive-risk lexicon** — 36 hand-curated regex patterns that capture risky wording ("sole discretion", "without limitation", "irrevocable", "perpetuity", etc.)

A **substantive-risk override** ensures that strong risky-language signals floor the final score — because risk is *substantive*, not just structural.

### Empirical Result

On a hand-curated benchmark of 5 textbook high-risk clauses vs. 5 textbook standard clauses:

| Group | Avg risk score |
|---|---|
| High-risk | **0.74** (all flagged as HIGH) |
| Standard | **0.15** (all flagged as LOW) |
| **Separation** | **0.59 gap** |

**100% accuracy on the benchmark.** The pipeline is fully deterministic — identical inputs yield identical outputs, byte-for-byte.

---

## 🏗️ Architecture
