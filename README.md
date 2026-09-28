# Contract Clause Risk Analyzer

A clause-level risk detection system for legal contracts — rental agreements,
employment contracts, vendor agreements, and more.

Unlike traditional tools that classify a **whole document** as risky, this system:

- Splits the contract into individual **clauses**
- Classifies each clause by **type** (termination, indemnity, etc.)
- Detects **unusual/risky clauses** with clause-level granularity
- Compares against a **reference clause bank** using semantic similarity
- Explains *why* a clause is risky in plain English (RAG + LLM)
- **Highlights** the exact text span that triggered each risk flag

## Research Angle

**Clause-level risk detection** using NER + clause classification + semantic
similarity + retrieval-augmented explanation. Positioned as a step beyond
document-level contract classifiers.

## Status

🚧 Under construction — Phase 0 (foundations).

## Tech Stack (planned)

- **Parsing:** PyMuPDF, python-docx
- **NLP:** spaCy, HuggingFace Transformers, Sentence-Transformers
- **Vector search:** FAISS
- **Backend:** FastAPI
- **Frontend:** React + Tailwind
- **LLM (explanation only):** Google Gemini API (swappable)

## Project Structure

    src/
      ingestion/       Document parsing (PDF/DOCX/TXT → clean text)
      segmentation/    Split contract into clauses with char offsets
      nlp/             NER + clause classification + embeddings
      risk/            Multi-signal risk scoring
      rag/             Retrieval + explanation generation
      api/             FastAPI backend
    config/            All settings (config.yaml) — no hardcoding
    data/              Raw, processed, reference data
    models/            Trained weights + FAISS index
    notebooks/         Experiments and Colab training
    tests/             Unit tests
    frontend/          React web app

## Setup

Coming soon (Phase 0, Step 4).

## License

For academic / research use.