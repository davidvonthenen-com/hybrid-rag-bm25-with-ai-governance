# **Document RAG for Better AI Governance**

## **1. Executive Summary**

Most RAG implementations lean on vectors alone or "hybrid" search that blends dense vectors with lexical scoring. Hybrid improves recall and robustness, and a [recent AWS write-up](https://aws.amazon.com/blogs/big-data/hybrid-search-with-amazon-opensearch-service) details how OpenSearch mixes BM25/lexical/full-text with vectors (and even sparse vectors) to boost retrieval quality. That said, [hybrid (even the hybrid version that exists natively with OpenSearch)](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/) still hides why a passage surfaced-semantic similarity scores aren't human-interpretable, and default lexical setups don't spell out which fields and clauses actually matched.

A domain-aware [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) first pipeline centers retrieval on explicit entities the domain cares about. In this repository, NER runs as an external HTTP service invoked by the Python ingester; its outputs are stored on each document as `explicit_terms` (keyword with lowercase normalizer) and `explicit_terms_text` (text). Queries are lexical-first BM25 and use explicit, auditable clauses: a strict **AND** via `terms_set` on `explicit_terms` (all detected entities must be present in the same doc), an **OR** branch that blends entity terms, and `multi_match` over `content` and `category`. We return highlights so humans can see the exact matching spans.

From an AI Governance standpoint, this design is superior on three fronts:

* **Transparency and Explainability**: Each answer is tied to specific fields and query branches (e.g., `terms_set` on `explicit_terms`, `multi_match` on `content`/`category`) with highlights, so you can show precisely why a document matched and how it scored.
* **Accountability and Responsibility**: Retrieval steps are reproducible and loggable. Documents carry `filepath`/`URI`, `ingested_at_ms`, and `doc_version`. If operators choose to materialize items across stores, they do so via a plain, reviewable `/_reindex` job-kept outside the query path and easy to audit. Promotion **from HOT to LT** happens only when there is (1) enough positive reinforcement of the data or (2) a trusted human-in-the-loop has verified the data.
* **Data Governance and Regulatory Compliance**: The dual-store layout is explicit. **HOT (unstable)** holds time-boxed, experiment-heavy material governed by TTL/rollover policies; **LT** retains vetted knowledge with provenance metadata. This makes retention and access policies enforceable and keeps audited content separate from unverified input.
* **Risk Management and Safety**: Answers are grounded in real documents and deterministic lexical logic. By requiring entity concurrence in the same doc and by exposing highlights, the system reduces hallucinations and makes stale or noisy hits easier to detect and remove.

This approach also benefits from recent research that marries symbolic structure with neural retrieval. For example, [HybridRAG (research originating from NVIDIA and Blackrock)](https://arxiv.org/pdf/2408.04948) (not to be confused with [OpenSearch's hybrid search functionality](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)) shows that explicit entity/relationship extraction feeding a structured store (e.g., this Document RAG method, a Knowledge Graph, or an entity index) improves precision and evidence quality because the system targets semantically relevant and structurally grounded facts. Additionally ,this Document RAG implementation mirrors capabilities produced by [Graph-based RAG and Knowledge Graphs](https://neo4j.com/blog/genai/what-is-graphrag/) but using Document search as the focal point.

**Bottom line:**
Pure vectors maximize fuzzy recall; hybrid (vector+BM25/sparse) balances fuzziness with keywords. Domain-trained NER + BM25 with entity-aware clauses makes the system explainable and governable without sacrificing retrieval quality. If your bar includes regulatory traceability, incident forensics, and reproducible outcomes (not only "good answers"), promote entities to first-class citizens (`explicit_terms`) and let vectors play a supporting role.

> **IMPORTANT NOTE:** This implementation uses OpenSearch with explicit, fielded lexical queries and an external NER service to enrich documents at ingest. Searches query **LT and HOT in parallel**, then (by default) a lightweight **external BM25 re-ranker** re-scores the combined hit list before prompts are built, preserving determinism and explainability. Highlight snippets and stable mappings keep retrieval observable and auditable while reducing hallucinations by grounding answers in verifiable documents. Promotion **from HOT to LT** is a controlled, reviewable step performed only with sufficient reinforcement or human verification.

## **2. Document-Based RAG Architecture**

### **High-Level Architecture Description**

At a high level, the Document RAG architecture consists of three main components:

1. **Large Language Model (LLM)**: Generates responses from retrieved context plus the user's question, and is constrained to that context.

2. **OpenSearch Knowledge Stores**: Two OpenSearch instances play distinct roles:

   * **Long-Term (LT)** holds the durable corpus. Documents are enriched at ingest via an external NER service and indexed with deterministic mappings, including `explicit_terms` (keyword, lowercase normalizer) and `explicit_terms_text` (text), plus provenance fields like `filepath`/`URI`, `ingested_at_ms`, and `doc_version`.

   * **HOT (unstable)** holds volatile RL/experimental data and user-defined data or facts. HOT uses permissive schemas and TTL/rollover.

3. **Integration Layer (Middleware)**: Connects the LLM and OpenSearch. For each question it calls the external NER API, builds an **auditable, entity-aware dis_max** query, and **queries LT and HOT in parallel**. This NER API is external and intentional. For maximum precision and correctness, it's highly recommended that a custom NER tuned to your problem set is created. After both stores respond, the middleware can optionally apply an **external BM25 re-ranker** (default behavior in this repo) to the merged hit list before collecting highlights and assembling the LLM prompt.

![Generic Document RAG Implementation](./images/reinforcement_learning.png)

In this implementation, OpenSearch is a **document search system**, not a black-box vector store. We rely on keyword matching, BM25 ranking, phrase constraints, and metadata filters (e.g., `explicit_terms`, `category`) to keep retrieval explainable and deterministic. Two instances exist primarily for **governance boundaries and retention variations**, not because latency alone demands it.

Overall, the design marries an LLM's generation with OpenSearch's transparent retrieval. You tune behavior by deciding what lives in LT vs HOT, which metadata you capture at ingest, and how the query branches are formulated. Next, we outline how the two stores work together to strengthen governance.

### **HOT vs. Long-Term Roles**

The architecture separates HOT and LT to optimize governance, provenance, and operational hygiene:

* **HOT (unstable)**: A store for **documents** that are experimental, user-generated, or otherwise unverified. HOT is optimized for write churn and fast reads, returns highlights for legibility, and is pruned by TTL/rollover (e.g., delete-by-query where `hot_promoted_at < now-TTL`). HOT is **not** a conversation log.

* **Long-Term (LT)**: The durable, vetted repository. Documents are ingested with NER enrichments (`explicit_terms`, `explicit_terms_text`) and provenance metadata. **Promotion from HOT → LT occurs only when** (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

### **Benefits of Document RAG**

Adopting a Document RAG architecture with OpenSearch brings several distinct advantages:

* **Structured Knowledge Representation**: Entities and provenance fields (`explicit_terms`, timestamps, categories, filepaths) give structure to unstructured text and enable precise, auditable filters.
* **Deterministic Retrieval**: The middleware issues entity-aware lexical queries to **both** stores in parallel, then (when enabled) uses an external BM25 stage to re-rank the merged hits so the LLM receives a single, auditable ordering without opaque score mixing.
* **Reduced Hallucinations, Improved Accuracy**: Answers are grounded in retrieved documents; strict entity concurrence (e.g., `terms_set` **AND**) reduces spurious hits.
* **Transparency and Traceability**: Highlights plus fielded clauses make it clear **why** a document matched and **which** fields contributed.
* **Open-Source Flexibility**: Built with OpenSearch, Flask, and Python-customizable and extensible without vendor lock-in.

In summary, Document RAG leverages transparent search-precision fields, clear analyzers, and explicit query branches-to build a governance-friendly RAG system. The next sections show how these choices increase explainability and how the system behaves in practice.

### **Enhancing Transparency and Explainability**

Transparency is built in and observable end-to-end:

* **Documented Evidence**: Every answer links back to specific documents. Highlights show the exact matched spans used as evidence.
* **Metadata Annotations**: NER outputs are indexed explicitly (`explicit_terms`, `explicit_terms_text`), so retrieval can be explained in human terms (which entities steered matching).
* **Explicit Query Logic**: The integration layer issues named/structured clauses (e.g., `terms_set` AND branches and entity-aware `multi_match`), targeting specific fields. This makes errors traceable and debugging feasible.
* **Audit Trails**: Provenance fields (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) and HOT stamps (e.g., `hot_promoted_at`) provide a clear trail from question → entities → per-store query plans → response. HOT → LT promotion events are discrete, reviewable steps.

Reasoning is externalized: we can map query → retrieved evidence → answer without relying on opaque similarity scores-useful for regulated domains where reviewers must see and verify the chain of custody.

### **Visualizing the Architecture (Referencing Diagram)**

To conceptualize this, picture two stores and an orchestrator:

* The **Orchestrator** receives a question, calls the **NER service**, builds an entity-aware dis_max query, and **queries LT and HOT in parallel**.
* It ranks **within each store**, optionally re-ranks the merged hits with an external BM25 step, and returns **highlights** with the final ordering.
* The **LLM** receives only the retrieved snippets as context and generates the answer.
* **Governance policy**: HOT → LT promotion happens **only** with sufficient positive reinforcement or explicit human verification.

TODO: Image

Unlike vector-only RAG, this dual-store, lexical-first design protects provenance and limits blast radius while keeping recent, entity-relevant content observable and auditable-without relying on hidden similarity scores or per-question data movement.

## **3. HOT (unstable) Store**

### **Overview of HOT**

HOT in this OpenSearch-based RAG system is a **document store**, not a chat transcript. It holds **volatile RL/experimental data** and, when desired, a **small, materialized subset** of long-term (LT) documents for operational reasons. Relevance for retrieval is driven by an external NER service that extracts entities from the user's question; the integration layer uses those entities to build **entity-aware BM25** queries and **queries LT and HOT in parallel**. When operators choose to materialize LT content into HOT, each copied document should be stamped with `hot_promoted_at` for later eviction.

This store is optimized for speed and legibility. Keeping HOT small means BM25 and phrase queries run fast and return **highlights** for inspection. The schema mirrors LT fields (`content`, `category`, `filepath`/`URI`, `explicit_terms`, `explicit_terms_text`, `ingested_at_ms`, `doc_version`) with an extra `hot_promoted_at` to support TTL. HOT is **self-pruning**: a scheduled eviction job deletes items older than a configured window, keeping the store lean and current. LT remains the source of truth.

### **Implementation Details**

Implementing HOT with OpenSearch centers on how documents are **materialized (optional)**, **queried**, and **evicted**:

* **Index Design and Schema**: Use deterministic mappings.
  `explicit_terms` as `keyword` with a lowercase normalizer; `explicit_terms_text` as `text`; preserve `content`, `category`, `filepath`/`URI`, `ingested_at_ms`, `doc_version`; add `hot_promoted_at` as `date` (epoch_millis). Favor `number_of_replicas: 0` and a tight `refresh_interval` for latency.

* **External NER, not ingest pipeline**: NER runs in a separate service (spaCy for the reference implementation). At **ingest time**, LT stores NER outputs (`explicit_terms`, `explicit_terms_text`). At **question time**, the integration layer calls NER, builds an auditable dis_max query, and **hits LT and HOT in parallel**.

* **Expiration and Removal**: Evict via a **delete-by-query TTL job**. Delete documents where `hot_promoted_at < now - TTL`. Control throughput with `max_docs`, `requests_per_second`, and `wait_for_completion`.

* **HOT → LT Promotion Policy**: Promotion **from HOT to LT** happens **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

* **Isolation and Scaling**: Run HOT and LT as separate instances/clusters. HOT can sit on faster storage with low replication; LT prioritizes durability and governance controls.

Follow these practices and HOT behaves like a **governable, entity-scoped store** for fast, explainable retrieval.

### **Performance Considerations and Optimization**

HOT must respond quickly under load:

* **Keep It Small**: Keep HOT effectively memory-resident. Favor zero replicas for latency; depend on LT for durability.
* **Indexing and Refresh**: A short `refresh_interval` balances freshness and throughput.
* **Sharding**: For small HOT indices, a single shard avoids scatter/gather overhead.
* **Query Shape**: Use entity-aware, **fielded** queries. A strict **AND** via `terms_set` on `explicit_terms` ensures all required entities co-occur in the same document; add phrase or `multi_match` branches for recall. Request highlights on `content` for legibility. Execute the **same query against LT and HOT in parallel**, then (optionally) let the external BM25 step re-rank the merged hits before handing them to the LLM.
* **Resource Allocation**: Size CPU and I/O for promotion bursts and question spikes.
* **Maintenance**: Run the TTL eviction job on a schedule. Keep HOT lean; fewer docs mean faster queries and cheaper merges.

Applied together, these optimizations keep HOT responsive (milliseconds-scale), even when questions arrive in bursts.

### **Benefits of HOT**

A HOT layer improves both operations and governance:

* **Low-Latency Serving:** Small working sets return entity-relevant documents fast, improving time-to-first-token.
* **Deterministic Hygiene:** TTL eviction cleans stale items automatically and keeps lifecycle auditable (`hot_promoted_at`).
* **Explainable Context:** Highlights and fielded clauses make it clear **why** a document was used.
* **Governance by Design:** LT holds provenance (`filepath`/`URI`, `ingested_at_ms`, `doc_version`); HOT adds `hot_promoted_at` and enforces the **HOT → LT promotion rule** (reinforcement or human verification only).

In essence, HOT is a fast, entity-scoped store that boosts responsiveness while preserving the integrity and auditability of the long-term repository. It adapts to what matters **now**, without sacrificing traceability.

## **4. Long-Term Memory**

### **Overview of Long-Term Memory**

Long-term memory is the persistent knowledge foundation of the Document RAG architecture. This is where the system's accumulated information, expected to remain relevant over time, is stored. In practice, long-term memory is one (or more) OpenSearch indices on the **LT** instance. Unlike **HOT (unstable)**, which is ephemeral, long-term memory contains data that doesn't expire on a timer-it stays until updated or removed deliberately.

Some characteristics of long-term memory:

* **It is comprehensive:** The store covers a wide range of documents (manuals, knowledge articles, books, historical records). For enterprise assistants this can include policies, product docs, FAQs, and industry literature-material that benefits from durable indexing and provenance.

* **It is structured for retrieval:** In this reference implementation we index **whole documents** and, by default, **paragraph-level chunks** with deterministic mappings. Each record carries `content` (text), `category` (keyword, lowercase normalizer), `filepath`/`URI` (stable `_id`), `explicit_terms` (keyword, lowercase normalizer), `explicit_terms_text` (text), `ingested_at_ms` (epoch_millis), and `doc_version` (long). This supports precise BM25, phrase constraints, and entity-aware filters. The community ingest script (`community_version/ingest.py`) materializes this by writing full documents to `bbc` and paragraph slices to `bbc-chunks`, so practitioners can inspect and extend paragraph-level retrieval without guessing about the storage layout.

* **It ensures consistency and accuracy:** The LT store is curated via a controlled ingest path that **enriches with external NER** at write time and assigns stable IDs plus `doc_version`. Updates are performed by re-ingest, keeping the corpus reproducible.

* **It provides historical context:** Long-term memory holds enduring documents and facts (not chat transcripts).

* **It scales technically:** OpenSearch scales horizontally (sharding/replication) and can split by thematic index if needed. As volume grows, LT absorbs millions of documents while keeping mappings stable for deterministic behavior.

* **It evolves with time:** Long-term doesn't mean static; new material is ingested and older content can be revised or removed. Version fields (`doc_version`) and timestamps (`ingested_at_ms`) support governance and replay.

In essence, long-term memory acts as the AI's body of record. It complements **HOT (unstable)** by providing stability, provenance, and breadth.

### **Integration with HOT**

The interaction between long-term and HOT is what gives the system its power:

* **During Query Processing:** The orchestrator extracts entities from the user's question (using NER), builds an **auditable entity-aware BM25 query**, and **queries LT and HOT in parallel**. It ranks **within each store**, then (by default) re-ranks the merged hits with an external BM25 stage before finalizing the context handed to the LLM.

* **Promotion from HOT → LT** occurs **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data. Long-term remains authoritative.

* **Data Consistency:** Answers must stay consistent with the canonical source. If re-materialized, HOT entries are overwritten from LT; conflicts resolve to LT as source of truth.

* **Multi-Store Search:** OpenSearch can query multiple indices. This implementation **does** combine LT and HOT by interleaving per-store top results without cross-normalizing scores, keeping behavior deterministic and observable.

* **Lifecycle & Consolidation:** Consolidation of knowledge happens through the ingest path (updating LT) rather than copying from HOT. HOT is routinely pruned and rebuilt on demand.

* **Feedback Loop:** Usage signals (which entities drive hits, near-misses, frequent queries) inform ingest priorities in LT and tuning of entity extraction and query branches.

### **Performance and Scalability Considerations**

Long-term memory contains most of the data, so scale and steadiness matter:

* **Scalability:** Use sharding/replication to spread load; organize by thematic indices when helpful. Keep analyzers/normalizers pinned for reproducibility.

* **Indexing Throughput:** Bulk operations with deferred refresh improve large ingests; for rolling updates, re-ingest by stable `_id` (`filepath`/`URI`) and bump `doc_version`.

* **Resource Management:** Set replicas for resilience and use durable storage. Tier colder indices to cheaper media as needed while keeping hot paths responsive.

* **Backup and Recovery:** Take regular snapshots (e.g., S3/Azure/NFS). Storage-level replication can add DR protection; verify restores against mappings and doc counts.

* **Monitoring and Optimization:** Track latency, heap, and segment counts. Add fields or indices to match query patterns; tune refresh/merge policy based on actual workload.

* **Security and Multitenancy:** Enforce role-based access; validate performance with security enabled. Document- or field-level controls are possible but add overhead-measure before and after.

Treat the long-term store like a production search service: stable mappings, capacity planning, and steady maintenance.

## **5. AI Governance Improvements**

Effective AI governance means ensuring that AI systems operate in a manner that is transparent, fair, accountable, and safe, while adhering to relevant laws and ethical standards. The Document RAG architecture we've described offers concrete improvements in each of these areas by design. Let's break down the governance benefits across several key dimensions:

### **Transparency and Explainability**

The system links each AI answer back to specific documents and fields. Retrieval is **lexical-first** (BM25, phrase queries) with explicit, auditable clauses against `content`, `category`, and entity fields (`explicit_terms`, `explicit_terms_text`). We return **highlights** so reviewers can see the exact matched spans. LT and **HOT (unstable)** are queried **in parallel**, then the combined hit list can be re-ranked with the external BM25 stage to keep score math observable and explainable.

### **Fairness and Bias Mitigation**

Fairness starts with curation of the long-term (LT) corpus and visibility into retrieved entities and categories. Because entity extraction and query branches are explicit, teams can audit which entities drive results and adjust sources or rules when skew appears. Search analytics over `category`, `explicit_terms`, and `filepath`/`URI` make source over-reliance measurable and correctable.

### **Accountability and Responsibility**

Every critical step is loggable in plain terms. The NER API returns a `request_id` and detected `entities`. The query orchestrator can emit observability summaries and optional JSONL records (`--save-results`) or be send to a 3rd party external database that include the question, entities, per-store totals, and the filepaths of kept hits. In other words, the transaction log. These artifacts trace question → entities → per-store queries → selected context → answer.

### **Data Governance**

Lifecycle control is built in. LT is durable and versioned; **HOT** is ephemeral and pruned by a **TTL eviction job** (`delete_by_query` where `hot_promoted_at < now - TTL`). Deterministic mappings (normalizers, field types) and stable IDs (`filepath`/`URI`) make schema validation and data hygiene operational, not aspirational. **HOT → LT promotion occurs only** when there is sufficient positive reinforcement **or** a trusted human-in-the-loop has verified the data.

### **Regulatory Compliance and Standards**

This design supports rights and controls that regulators care about. Source traceability and fielded queries enable evidence production; precise deletes against LT handle erasure requests; HOT TTL prevents transient copies from lingering. Access control and residency are enforced at the OpenSearch layer and can be audited alongside search activity. Snapshots taken on LT provide point-in-time attestations of "what the system knew."

### **Risk Management and Safety**

Grounded, fielded retrieval reduces hallucinations. The **entity concurrence** requirement (e.g., `terms_set` AND on `explicit_terms`) lowers the chance of spurious matches. Querying LT and HOT in parallel with per-store ranking prevents cross-store score drift. HOT's TTL curbs stale context. When issues occur, audit artifacts (NER `request_id`, saved JSONL, highlights, and OpenSearch job logs) speed root-cause analysis and rollback.

Document RAG for AI governance takes the mystery out of the machine. Two OpenSearch instances exist primarily for **governance boundaries and retention variations control**, not for latency wins-so performance and responsibility move forward together.

## **6. Target Audience and Use Cases**

Document RAG with OpenSearch is a flexible architecture that serves multiple stakeholders. Below we outline the primary audiences and concrete use cases, aligned with the **dual-store, lexical-first** design: an external NER service, **entity-aware BM25 queries run against LT and HOT in parallel**, deterministic per-store ranking, and interleaving of results. **HOT → LT promotion occurs only** with sufficient positive reinforcement **or** trusted human verification.

### **Open-Source Engineers**

Builders who value transparency, composability, and zero vendor lock-in.

* **Why it matters**
  Everything is inspectable: deterministic mappings (`explicit_terms`, `explicit_terms_text`, provenance fields), an external NER service, **no vectors required**, and a clear dual-store plan. Queries are auditable (`terms_set` AND, phrase/multi-match), and results include highlights.

* **Extensible data modeling**
  Swap in domain NER without changing the retrieval contract. Keep shared schemas stable while evolving analyzers and fielded clauses. Observability flags (e.g., saved JSONL) make experiments repeatable.

* **Use case**
  A programming Q&A assistant ingests manuals, API docs, and forum answers into **LT** with NER enrichments. At question time, NER extracts entities (APIs, error codes); the orchestrator **queries LT and HOT in parallel**, re-ranks the merged hits with the external BM25 stage, and sends highlights to the LLM. A nightly job **materializes** popular LT docs into **HOT** via `/_reindex`; TTL cleans them up. No per-question copying, no mystery scores.

### **Enterprise Architects**

Leaders who must integrate AI into existing estates with guardrails for scale, security, and compliance.

* **Why it matters**
  Deterministic mappings, **entity-aware lexical queries**, and **parallel LT+HOT search** keep behavior predictable.

* **Governance & compliance**
  Answers trace to documents and fields. Provenance (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) plus optional `hot_promoted_at` on materialized items support audits. **HOT → LT promotion** is gated by reinforcement or human sign-off.

* **Scalable infrastructure**
  Run **LT** for durability and **HOT** for volatile/experimental workloads with different SLAs. Capacity planning follows standard OpenSearch practices; snapshots on LT provide point-in-time attestations.

* **Use case**
  A policy assistant for a financial firm ingests manuals and memos into **LT**. Queries hit **LT and HOT** in parallel; the merged hit list is optionally re-ranked externally with BM25 before being highlighted. Everything runs in-VPC with enterprise auth.

### **NetApp Infrastructure Customers**

Teams standardizing on ONTAP who want storage-level reliability with application-level transparency.

* **Why it matters**
  The architecture cleanly maps to storage controls: **LT** on durable tiers with snapshots/replication, **HOT** on low-latency media with low replication and TTL. The app path stays simple and auditable.

* **Performance on NetApp**
  Put **HOT** on faster storage to reduce tail latencies for BM25/phrase queries; use caching (e.g., FlexCache) to keep hot index segments close to compute.

* **Use case**
  A support assistant indexes product manuals and KBs into **LT**. Queries run against **LT and HOT** in parallel. SnapMirror feeds a DR site and dev/test.

### **Cross-Industry Applicability**

The pattern stays the same; only the corpus and NER/BM25 implementation changes.

* **Healthcare**
  Clinical guidelines and formularies live in **LT** with provenance. Queries run against **LT and HOT**; highlights show exact evidence. On-prem deployments align with HIPAA controls.

* **Retail & E-commerce**
  Product specs, return policies, and compatibility matrices sit in **LT**. Entity-aware queries retrieve precise answers; popular seasonal content can be materialized into **HOT** on a schedule and evicted by TTL.

* **Legal & Regulatory**
  Statutes, rulings, and memos are ingested into **LT** with stable IDs. Fielded, explainable matches power counsel workflows; any HOT presence is operational (materialized windows), not a hidden data path.

Across audiences, the benefits are consistent: **explainability, determinism, and operational control**. The dual-store design proves where answers came from, separates durable truth from volatile experiments, and turns policy enforcement into configuration-while keeping the request path fast and observable.

## **7. Implementation Guide**

Please see [GitHub repository](https://github.com/davidvonthenen/docuemnt-rag-guide) for:

- [Community / Open Source Implementation](./OSS_Community_Implementation.md)
 For a reference implementation, please check out the following: [community_implementation/README.md](./community_implementation/README.md)

- [Enterprise Implementation](./Enterprise_Implementation.md)
 For a reference implementation, please check out the following: [enterprise_implementation/README.md](./enterprise_implementation/README.md)

## **8. Conclusion**

The Document RAG architecture moves AI retrieval from opaque heuristics to **observable, governable search**. Pairing a Large Language Model with **lexical-first OpenSearch retrieval** (BM25, phrase constraints, and fielded entity clauses) blends generation with verifiable evidence. Queries run against **both Long-Term (LT) and HOT (unstable)** in parallel, and (by default) the merged hit list is re-ranked externally with BM25 before final delivery—maintaining reliability, transparency, and compliance that end-to-end training or vector-only stacks can't match.

Knowledge is treated as a first-class asset. **LT (LT)** is the vetted, durable store with deterministic mappings (`explicit_terms`, provenance fields, versions). **HOT (unstable)** is an operational, entity-scoped store governed by TTL. **HOT → LT promotion happens only** when there is sufficient **positive reinforcement** of the data **or** a **trusted human-in-the-loop** has verified it.

Transparency is built in. Answers are grounded in retrievable documents with **highlights** and auditable query branches (e.g., `terms_set` AND on `explicit_terms`, `multi_match`/phrase on `content`/`category`). The orchestrator's observability controls (per-store totals, kept filepaths, and optional compact JSONL records) make the path from **question → entities → per-store results → answer** explainable and reproducible for reviewers and auditors.

Finally, this architecture aligns with enterprise governance. Benchmarks show **latency isn't the primary reason** to split stores; two OpenSearch instances exist for **governance boundaries, retention variations control, and policy asymmetry**. Using **deterministic analyzers, explicit metadata, and TTL eviction**, the system meets accountability and regulatory needs without LTing delivery. Built on mature, open-source tools (OpenSearch, Flask, Python), it's practical, scalable, and cost-effective. Document RAG proves powerful AI can be both **capable and accountable**.

