# Hybrid BM25 + Vector RAG (Community Edition)

This repository contains a laptop-friendly hybrid Retrieval-Augmented Generation (RAG) stack built on **OpenSearch BM25**, optional **vector search**, and a local **llama.cpp** model. The Python code under `community_version/` is the authoritative implementation; documentation now mirrors that code.

The system favors explainability and auditability:

* **Lexical-first retrieval.** Queries start with entity-aware BM25 searches against paragraph-level chunks in both long-term and HOT stores.
* **Optional dense signals.** Question embeddings can add vector matches, but lexical evidence always remains in the result set.
* **Transparent scoring.** An external BM25 re-ranker (via [`bm25s`](https://github.com/xhluca/bm25s)) is enabled by default so the ranked list is reproducible and inspectable.

## Repository layout

* `community_version/` – runnable hybrid RAG demo (ingestion, query, HOT reinforcement utilities).
* `Document_Search_for_Better_AI_Governance.md` – background on why lexical-first retrieval improves governance.
* `OSS_Community_Version.md` / `Enterprise_Version.md` – narrative guides for broader deployments.

If you want to run the stack, start with `community_version/README.md` for setup, configuration, and commands.

## Quick start

```bash
cd community_version
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the local spaCy NER API required by the pipeline
python ner_service.py

# Ingest the BBC dataset into OpenSearch (long-term BM25 + vector chunks)
python ingest.py --data-dir bbc

# Ask questions using hybrid BM25 + vector retrieval
python query.py --question "What did Google purchase Windsurf for?" --observability
```

See the community README for environment variable overrides (OpenSearch hosts/indices, llama.cpp model path, ranking knobs) and HOT memory workflows.

