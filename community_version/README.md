# Community Hybrid RAG

The `community_version` folder contains a self-contained hybrid RAG stack that combines:

* **Paragraph-level BM25 search** over long-term and HOT OpenSearch indices with NER-derived entity fields.
* **Optional k-NN vector search** over chunk embeddings.
* **Local llama.cpp inference** with a constrained prompt so answers cite only retrieved context.

The Python code is the source of truth for behavior; this README mirrors the current implementation.

## Components

* `ingest.py` – walks the BBC news dataset, extracts named entities, and indexes both full documents and paragraph chunks into BM25 plus a vector chunk index.
* `query.py` – CLI that runs NER on the question, queries LONG + HOT + vector stores in parallel, optionally re-ranks with the local `bm25s` pass, and feeds the merged context to the LLM.
* `example/reinforcement_learning.py` – populates the HOT index with demo facts and exercises the hybrid query path.
* `expire_hot_data.py` / `manual_promote.py` – helpers for expiring or promoting HOT facts when experimenting with governance flows.
* `ner_service.py` – lightweight Flask HTTP service that exposes spaCy NER (`/health`, `/ner`). All ingestion and query paths depend on it.

## Prerequisites

* Python 3.10+
* Two reachable OpenSearch nodes (or two indices on one node) for LONG and HOT. Defaults: `127.0.0.1:9201` (LONG) and `127.0.0.1:9202` (HOT).
* Local llama.cpp-compatible model file (path configurable via `LLAMA_MODEL_PATH`).
* BBC dataset extracted under `community_version/bbc/<category>/*.txt` (default `--data-dir bbc`).
* spaCy model `en_core_web_sm` installed for NER.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the HTTP NER service used by ingest/query
python ner_service.py
```

Ensure your OpenSearch instances are running and reachable according to the environment variables below before ingesting data.

## Data ingestion

```bash
python ingest.py --data-dir bbc --batch-size 32
```

What happens during ingestion (per the code in `ingest.py`):

1. Connects to the LONG OpenSearch host and ensures three indices exist using the configured names:
   * Full-document BM25: `opensearch_full_index` (default `bbc-bm25-full`).
   * Paragraph BM25 chunks: `opensearch_long_index` (default `bbc-bm25-chunks`).
   * Vector chunk index: `opensearch_vector_index` (default `bbc-vector-chunks`).
2. Splits each BBC article into paragraphs, extracts entities via the running NER service, and writes:
   * Full documents with `explicit_terms`/`explicit_terms_text` metadata to the full BM25 index.
   * Paragraph chunks with chunk metadata (`chunk_index`, `chunk_count`, `parent_filepath`) to the BM25 chunk index.
   * Chunk embeddings (using `thenlper/gte-small` by default) to the vector index.
3. Refreshes all indices and logs counts of documents and chunks ingested.

The ingest script only targets the LONG cluster; HOT content is added separately via the reinforcement or promotion scripts.

## Querying

```bash
python query.py --question "How much did Google purchase Windsurf for?" --observability --top-k 8
```

Query flow (implemented in `common/llm.py`):

1. Run NER over the question; build an entity-aware BM25 query using `terms_set` + `multi_match` branches.
2. Execute LONG BM25, HOT BM25, and vector k-NN searches **in parallel**.
3. By default, re-rank the combined LONG+HOT BM25 hits locally with `bm25s`; otherwise rely on OpenSearch scores (`--observability` prints the choice).
4. Normalize vector hits and merge them with BM25 results, reserving a configurable share for dense matches via `RANKING_ALPHA`.
5. Build a provenance-friendly context block (store label + filepath + content) and pass it to the llama.cpp model to generate the answer.
6. Optional `--save-results <path>` writes JSONL audit records of the run.

If `--question` is omitted, the script uses two built-in sample questions. The CLI prints query JSON, store summaries, and kept matches when `--observability` is enabled.

## HOT memory workflow

* `example/reinforcement_learning.py` prompts you to inject example facts into the HOT index, then asks a few test questions through the same query path.
* `manual_promote.py` marks HOT facts for promotion, and `expire_hot_data.py` removes expired items to simulate cache cleanup.

These helpers rely on the same field schema as the LONG BM25 chunk index. Set `OPENSEARCH_HOT_*` variables if your HOT store differs from the defaults.

## Configuration

All scripts load settings from `common/config.py` with `.env` support. Key overrides:

| Purpose | Environment variables | Defaults |
| --- | --- | --- |
| LONG BM25 host/index | `OPENSEARCH_LONG_HOST`, `OPENSEARCH_LONG_PORT`, `OPENSEARCH_LONG_INDEX` | `127.0.0.1`, `9201`, `bbc-bm25-chunks` |
| HOT BM25 host/index | `OPENSEARCH_HOT_HOST`, `OPENSEARCH_HOT_PORT`, `OPENSEARCH_HOT_INDEX` | `127.0.0.1`, `9202`, `bbc-bm25-chunks` |
| Full-document index | `OPENSEARCH_FULL_INDEX` | `bbc-bm25-full` |
| Vector index | `OPENSEARCH_VECTOR_INDEX` | `bbc-vector-chunks` |
| Search sizing & ranking | `SEARCH_SIZE`, `RANKING_ALPHA`, `RAG_TOP_K`, `RAG_NUM_CANDIDATES` | `10`, `0.5`, `3`, `50` |
| LLM (llama.cpp) | `LLAMA_MODEL_PATH`, `LLAMA_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`, `LLAMA_N_BATCH`, `LLAMA_N_UBATCH`, `LLAMA_LOW_VRAM` | see defaults in `common/config.py` |
| NER service | `NER_URL`, `NER_TIMEOUT_SECS` | `http://127.0.0.1:8000/ner`, `5.0` |

The `RANKING_ALPHA` value is used both to threshold BM25 hits when OpenSearch scores are trusted and to determine how many vector results are reserved in the hybrid merge.

## Troubleshooting

* Verify the NER service is running; both ingest and query call it before constructing documents or queries.
* Ensure LONG and HOT hosts/ports match your OpenSearch deployment; connection errors surface in the CLI output when `--observability` is enabled.
* For Metal/low-VRAM environments, set `LLAMA_LOW_VRAM=true` and adjust `LLAMA_N_GPU_LAYERS` as needed.

