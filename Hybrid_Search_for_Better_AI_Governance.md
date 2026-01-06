# **Hybrid Search for Better AI Governance**

## **1. Executive Summary**

Most RAG implementations lean on vectors alone or on generic “hybrid” search that blends dense vectors with lexical scoring. Hybrid improves recall and robustness, and a [recent AWS write-up](https://aws.amazon.com/blogs/big-data/hybrid-search-with-amazon-opensearch-service) details how OpenSearch mixes BM25/lexical/full-text with vectors (and even sparse vectors) to boost retrieval quality. That said, hybrid systems often blur why a passage surfaced—semantic similarity scores aren’t human-interpretable, and default lexical setups don’t always make it clear which fields or clauses matched.

This repository implements a **Hybrid RAG** that is intentionally **two-channel**:

* **BM25 as the grounding channel** for factual evidence and auditability.
* **Vector kNN as the semantic support channel** to add contextual language and phrasing without introducing new facts.

A domain-aware [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) service runs at ingest and query time. At ingest, entities are stored as `explicit_terms` and `explicit_terms_text` alongside content and provenance metadata. At query time, entities bias BM25 retrieval and help anchor vector search to relevant documents. The vector search is **filtered to BM25-anchored documents whenever possible**, which reduces semantic drift while still capturing useful paraphrases and context.

From an AI Governance standpoint, this design is superior on four fronts:

* **Transparency and Explainability**: Factual claims are grounded in BM25 evidence first, with explicit references to which documents contributed to the answer. Vector evidence is used only to clarify, not to introduce new facts.
* **Accountability and Responsibility**: Retrieval steps are reproducible and loggable. Documents carry `filepath`/`URI`, `ingested_at_ms`, and `doc_version`. The query runner can emit full observability traces and JSONL audit records that capture queries, retrieved chunks, and final answers.
* **Data Governance and Regulatory Compliance**: The dual-store layout is explicit. **HOT (unstable)** holds user-specific or experimental material governed by TTL/rollover policies; **LT** retains vetted knowledge with provenance metadata. This makes retention and access policies enforceable and keeps audited content separate from unverified input.
* **Risk Management and Safety**: Answers are grounded in retrieved documents and deterministic lexical logic. Vector context is constrained to BM25 anchors, reducing hallucinations and making noise easier to detect.

This approach benefits from recent research that marries symbolic structure with neural retrieval. For example, [HybridRAG (research originating from NVIDIA and Blackrock)](https://arxiv.org/pdf/2408.04948) (not to be confused with [OpenSearch’s hybrid search functionality](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)) shows that explicit entity/relationship extraction feeding a structured store improves precision and evidence quality. This Hybrid RAG implementation mirrors those goals by using entity-aware BM25 for factual grounding and vectors for semantic support.

**Bottom line:**
Pure vectors maximize fuzzy recall; hybrid (vector+BM25) balances fuzziness with keywords. **This repo goes further by separating the roles**—BM25 provides the evidence, vectors provide the phrasing—so answers stay explainable, reproducible, and governable without sacrificing retrieval quality.

> **IMPORTANT NOTE:** This implementation uses OpenSearch with explicit BM25 queries, external NER enrichment, and a dedicated vector index. Searches query **LT and HOT in parallel** for BM25 evidence, then use **vector kNN** for semantic context, typically filtered to BM25-anchored documents. The LLM first drafts a grounded answer from BM25 evidence and then (optionally) refines language using vector context without adding new facts. This preserves determinism and auditability while improving readability.

## **2. Document-Based RAG Architecture**

### **High-Level Architecture Description**

At a high level, the Hybrid RAG architecture consists of three main components:

1. **Large Language Model (LLM)**: Generates responses from retrieved context plus the user’s question, and is constrained to that context.

2. **OpenSearch Knowledge Stores**: Two OpenSearch instances play distinct roles:

   * **Long-Term (LT)** holds the durable corpus. Documents are enriched at ingest via an external NER service and indexed with deterministic mappings, including `explicit_terms` and `explicit_terms_text`, plus provenance fields like `filepath`/`URI`, `ingested_at_ms`, and `doc_version`.

   * **HOT (unstable)** holds volatile, user-specific, or experimental data. HOT uses permissive schemas and TTL/rollover policies.

   * **Vector Index** holds paragraph-level embeddings derived from the same corpus. This index is used for semantic kNN retrieval and is typically filtered to BM25-anchored documents.

3. **Integration Layer (Middleware)**: Connects the LLM and OpenSearch. For each question it calls the external NER API, builds an **auditable BM25 query**, and **queries LT and HOT in parallel**. It also runs a **vector kNN search** against the embedding index, filtered to BM25 anchors when available. The middleware then prepares LLM prompts that keep BM25 as the authoritative evidence and vectors as optional semantic support.

![Generic Hybrid RAG Implementation](./images/reinforcement_learning.png)

In this implementation, OpenSearch is a **document search system**, not a black-box vector store. We rely on fielded BM25 queries, entity-biased matching, and provenance metadata to keep retrieval explainable and deterministic. Vectors augment the system by improving semantic recall, but factual grounding remains lexical-first.

Overall, the design marries an LLM’s generation with transparent retrieval. You tune behavior by deciding what lives in LT vs HOT, how NER enrichments are stored, and how vector retrieval is constrained. Next, we outline how the two stores work together to strengthen governance.

### **HOT vs. Long-Term Roles**

The architecture separates HOT and LT to optimize governance, provenance, and operational hygiene:

* **HOT (unstable)**: A store for **documents** that are experimental, user-generated, or otherwise unverified. HOT is optimized for write churn and fast reads, and is pruned by TTL/rollover. HOT is **not** a conversation log.

* **Long-Term (LT)**: The durable, vetted repository. Documents are ingested with NER enrichments (`explicit_terms`, `explicit_terms_text`) and provenance metadata. **Promotion from HOT → LT occurs only when** (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

### **Benefits of Hybrid RAG**

Adopting this Hybrid RAG architecture with OpenSearch brings several distinct advantages:

* **Structured Knowledge Representation**: Entities and provenance fields (`explicit_terms`, timestamps, categories, filepaths) give structure to unstructured text and enable precise, auditable filters.
* **Deterministic Retrieval**: BM25 evidence is retrieved from LT and HOT in parallel using explicit query logic. Vector results are anchored to BM25 hits when possible to keep semantic context aligned with factual evidence.
* **Reduced Hallucinations, Improved Accuracy**: Answers are grounded in BM25 evidence; vectors are used only for clarifications or phrasing support.
* **Transparency and Traceability**: Observability tooling can emit full queries, per-store results, and final answers for audit purposes.
* **Open-Source Flexibility**: Built with OpenSearch, Flask, and Python; fully customizable and extensible without vendor lock-in.

In summary, this Hybrid RAG approach combines transparent search with semantic augmentation to deliver governance-friendly answers. The next sections show how these choices increase explainability and how the system behaves in practice.

### **Enhancing Transparency and Explainability**

Transparency is built in and observable end-to-end:

* **Documented Evidence**: Every answer links back to specific documents. BM25 evidence is always the authoritative source for factual claims.
* **Metadata Annotations**: NER outputs are indexed explicitly (`explicit_terms`, `explicit_terms_text`), so retrieval can be explained in human terms.
* **Explicit Query Logic**: The integration layer issues structured BM25 queries with entity-biased clauses and deterministic field selection.
* **Audit Trails**: Provenance fields (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) and HOT stamps provide a clear trail from question → entities → per-store retrieval → answer. HOT → LT promotion events are discrete, reviewable steps.

Reasoning is externalized: we can map query → retrieved evidence → answer without relying on opaque similarity scores—useful for regulated domains where reviewers must see and verify the chain of custody.

### **Visualizing the Architecture (Referencing Diagram)**

To conceptualize this, picture two stores and an orchestrator:

* The **Orchestrator** receives a question, calls the **NER service**, builds a BM25 query, and **queries LT and HOT in parallel**.
* It also runs **vector kNN** against the embedding index, filtering to BM25-anchored documents where possible.
* The **LLM** receives BM25 evidence as the authoritative grounding context, and vector context as optional semantic support.
* **Governance policy**: HOT → LT promotion happens **only** with sufficient positive reinforcement or explicit human verification.

TODO: Image

Unlike vector-only RAG, this dual-store, lexical-first design protects provenance and limits blast radius while keeping semantic augmentation constrained and auditable.

## **3. HOT (unstable) Store**

### **Overview of HOT**

HOT in this OpenSearch-based RAG system is a **document store**, not a chat transcript. It holds **volatile, user-specific, or experimental data** and, when desired, a **small, materialized subset** of long-term (LT) documents for operational reasons. Relevance for retrieval is driven by an external NER service that extracts entities from the user’s question; the integration layer uses those entities to build **entity-aware BM25** queries and **queries LT and HOT in parallel**. When operators choose to materialize LT content into HOT, each copied document should be stamped with `hot_promoted_at` for later eviction.

This store is optimized for speed and legibility. Keeping HOT small means BM25 queries run fast and return relevant chunks for inspection. The schema mirrors LT fields (`content`, `category`, `filepath`/`URI`, `explicit_terms`, `explicit_terms_text`, `ingested_at_ms`, `doc_version`) with an extra `hot_promoted_at` to support TTL. HOT is **self-pruning**: a scheduled eviction job deletes items older than a configured window, keeping the store lean and current. LT remains the source of truth.

### **Implementation Details**

Implementing HOT with OpenSearch centers on how documents are **materialized (optional)**, **queried**, and **evicted**:

* **Index Design and Schema**: Use deterministic mappings. `explicit_terms` as `keyword` with a lowercase normalizer; `explicit_terms_text` as `text`; preserve `content`, `category`, `filepath`/`URI`, `ingested_at_ms`, `doc_version`; add `hot_promoted_at` as `date` (epoch_millis). Favor low replicas and a tight `refresh_interval` for latency.

* **External NER, not ingest pipeline**: NER runs in a separate service (spaCy for the reference implementation). At **ingest time**, LT stores NER outputs (`explicit_terms`, `explicit_terms_text`). At **question time**, the integration layer calls NER, builds an auditable BM25 query, and **hits LT and HOT in parallel**.

* **Expiration and Removal**: Evict via a **delete-by-query TTL job**. Delete documents where `hot_promoted_at < now - TTL`. Control throughput with `max_docs`, `requests_per_second`, and `wait_for_completion`.

* **HOT → LT Promotion Policy**: Promotion **from HOT to LT** happens **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

* **Isolation and Scaling**: Run HOT and LT as separate instances/clusters. HOT can sit on faster storage with low replication; LT prioritizes durability and governance controls.

Follow these practices and HOT behaves like a **governable, entity-scoped store** for fast, explainable retrieval.

### **Performance Considerations and Optimization**

HOT must respond quickly under load:

* **Keep It Small**: Keep HOT effectively memory-resident. Favor low replicas for latency; depend on LT for durability.
* **Indexing and Refresh**: A short `refresh_interval` balances freshness and throughput.
* **Sharding**: For small HOT indices, a single shard avoids scatter/gather overhead.
* **Query Shape**: Use entity-aware, **fielded** queries. Strict BM25 settings can bias toward entity overlap while still permitting recall. Execute the **same BM25 query against LT and HOT in parallel** so evidence stays comparable.
* **Resource Allocation**: Size CPU and I/O for promotion bursts and question spikes.
* **Maintenance**: Run the TTL eviction job on a schedule. Keep HOT lean; fewer docs mean faster queries and cheaper merges.

Applied together, these optimizations keep HOT responsive (milliseconds-scale), even when questions arrive in bursts.

### **Benefits of HOT**

A HOT layer improves both operations and governance:

* **Low-Latency Serving:** Small working sets return entity-relevant documents fast, improving time-to-first-token.
* **Deterministic Hygiene:** TTL eviction cleans stale items automatically and keeps lifecycle auditable (`hot_promoted_at`).
* **Explainable Context:** Fielded clauses and provenance metadata make it clear **why** a document was used.
* **Governance by Design:** LT holds provenance (`filepath`/`URI`, `ingested_at_ms`, `doc_version`); HOT adds `hot_promoted_at` and enforces the **HOT → LT promotion rule** (reinforcement or human verification only).

In essence, HOT is a fast, entity-scoped store that boosts responsiveness while preserving the integrity and auditability of the long-term repository. It adapts to what matters **now**, without sacrificing traceability.

## **4. Long-Term Memory**

### **Overview of Long-Term Memory**

Long-term memory is the persistent knowledge foundation of the Hybrid RAG architecture. This is where the system’s accumulated information, expected to remain relevant over time, is stored. In practice, long-term memory is one (or more) OpenSearch indices on the **LT** instance. Unlike **HOT (unstable)**, which is ephemeral, long-term memory contains data that doesn’t expire on a timer—it stays until updated or removed deliberately.

Some characteristics of long-term memory:

* **It is comprehensive:** The store covers a wide range of documents (manuals, knowledge articles, books, historical records). For enterprise assistants this can include policies, product docs, FAQs, and industry literature—material that benefits from durable indexing and provenance.

* **It is structured for retrieval:** In this reference implementation we index **whole documents** and **paragraph-level chunks** with deterministic mappings. Each record carries `content` (text), `category` (keyword, lowercase normalizer), `filepath`/`URI` (stable `_id`), `explicit_terms` (keyword, lowercase normalizer), `explicit_terms_text` (text), `ingested_at_ms` (epoch_millis), and `doc_version` (long). This supports precise BM25 retrieval and entity-aware filters. Paragraph chunks are also embedded and stored in a vector index for semantic kNN retrieval.

* **It ensures consistency and accuracy:** The LT store is curated via a controlled ingest path that **enriches with external NER** at write time and assigns stable IDs plus `doc_version`. Updates are performed by re-ingest, keeping the corpus reproducible.

* **It provides historical context:** Long-term memory holds enduring documents and facts (not chat transcripts).

* **It scales technically:** OpenSearch scales horizontally (sharding/replication) and can split by thematic index if needed. As volume grows, LT absorbs millions of documents while keeping mappings stable for deterministic behavior.

* **It evolves with time:** Long-term doesn’t mean static; new material is ingested and older content can be revised or removed. Version fields (`doc_version`) and timestamps (`ingested_at_ms`) support governance and replay.

In essence, long-term memory acts as the AI’s body of record. It complements **HOT (unstable)** by providing stability, provenance, and breadth.

### **Integration with HOT**

The interaction between long-term and HOT is what gives the system its power:

* **During Query Processing:** The orchestrator extracts entities from the user’s question (using NER), builds an **auditable BM25 query**, and **queries LT and HOT in parallel**. It then runs **vector kNN** against the embedding index, typically filtered to BM25-anchored documents, and prepares LLM prompts with clear separation between grounding evidence and semantic support.

* **Promotion from HOT → LT** occurs **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data. Long-term remains authoritative.

* **Data Consistency:** Answers must stay consistent with the canonical source. If re-materialized, HOT entries are overwritten from LT; conflicts resolve to LT as source of truth.

* **Multi-Store Search:** OpenSearch can query multiple indices. This implementation **does** combine LT and HOT by interleaving per-store top results while maintaining deterministic retrieval and explicit separation between BM25 and vector channels.

* **Lifecycle & Consolidation:** Consolidation of knowledge happens through the ingest path (updating LT) rather than copying from HOT. HOT is routinely pruned and rebuilt on demand.

* **Feedback Loop:** Usage signals (which entities drive hits, near-misses, frequent queries) inform ingest priorities in LT and tuning of entity extraction and query biasing.

### **Performance and Scalability Considerations**

Long-term memory contains most of the data, so scale and steadiness matter:

* **Scalability:** Use sharding/replication to spread load; organize by thematic indices when helpful. Keep analyzers/normalizers pinned for reproducibility.

* **Indexing Throughput:** Bulk operations with deferred refresh improve large ingests; for rolling updates, re-ingest by stable `_id` (`filepath`/`URI`) and bump `doc_version`.

* **Resource Management:** Set replicas for resilience and use durable storage. Tier colder indices to cheaper media as needed while keeping hot paths responsive.

* **Backup and Recovery:** Take regular snapshots (e.g., S3/Azure/NFS). Storage-level replication can add DR protection; verify restores against mappings and doc counts.

* **Monitoring and Optimization:** Track latency, heap, and segment counts. Add fields or indices to match query patterns; tune refresh/merge policy based on actual workload.

* **Security and Multitenancy:** Enforce role-based access; validate performance with security enabled. Document- or field-level controls are possible but add overhead—measure before and after.

Treat the long-term store like a production search service: stable mappings, capacity planning, and steady maintenance.

## **5. AI Governance Improvements**

Effective AI governance means ensuring that AI systems operate in a manner that is transparent, fair, accountable, and safe, while adhering to relevant laws and ethical standards. The Hybrid RAG architecture we’ve described offers concrete improvements in each of these areas by design. Let’s break down the governance benefits across several key dimensions:

### **Transparency and Explainability**

The system links each AI answer back to specific documents and fields. Retrieval is **BM25-first** with explicit, auditable clauses against `content`, `category`, and entity fields (`explicit_terms`, `explicit_terms_text`). LT and **HOT (unstable)** are queried **in parallel**, and vector retrieval is **anchored to BM25 evidence** whenever possible. This preserves explainability while still benefiting from semantic context.

### **Fairness and Bias Mitigation**

Fairness starts with curation of the long-term (LT) corpus and visibility into retrieved entities and categories. Because entity extraction and query branches are explicit, teams can audit which entities drive results and adjust sources or rules when skew appears. Search analytics over `category`, `explicit_terms`, and `filepath`/`URI` make source over-reliance measurable and correctable.

### **Accountability and Responsibility**

Every critical step is loggable in plain terms. The NER API returns detected `entities`. The query orchestrator can emit observability summaries and optional JSONL records (`--save-results`) that include the question, entities, per-store totals, and the filepaths of kept hits—effectively the transaction log. These artifacts trace question → entities → per-store queries → selected context → answer.

### **Data Governance**

Lifecycle control is built in. LT is durable and versioned; **HOT** is ephemeral and pruned by a **TTL eviction job** (`delete_by_query` where `hot_promoted_at < now - TTL`). Deterministic mappings (normalizers, field types) and stable IDs (`filepath`/`URI`) make schema validation and data hygiene operational, not aspirational. **HOT → LT promotion occurs only** when there is sufficient positive reinforcement **or** a trusted human-in-the-loop has verified the data.

### **Regulatory Compliance and Standards**

This design supports rights and controls that regulators care about. Source traceability and fielded queries enable evidence production; precise deletes against LT handle erasure requests; HOT TTL prevents transient copies from lingering. Access control and residency are enforced at the OpenSearch layer and can be audited alongside search activity. Snapshots taken on LT provide point-in-time attestations of “what the system knew.”

### **Risk Management and Safety**

Grounded, fielded retrieval reduces hallucinations. BM25 evidence is the factual anchor, and vector context is constrained to that evidence whenever possible. HOT’s TTL curbs stale context. When issues occur, audit artifacts (entity extraction logs, saved JSONL, and OpenSearch traces) speed root-cause analysis and rollback.

Hybrid Search for AI governance takes the mystery out of the machine. Two OpenSearch instances exist primarily for **governance boundaries and retention variations control**, not for latency wins—so performance and responsibility move forward together.

## **6. Target Audience and Use Cases**

Hybrid RAG with OpenSearch is a flexible architecture that serves multiple stakeholders. Below we outline the primary audiences and concrete use cases, aligned with the **dual-store, BM25-grounded, vector-augmented** design: an external NER service, **entity-aware BM25 queries run against LT and HOT in parallel**, deterministic retrieval, and anchored vector augmentation. **HOT → LT promotion occurs only** with sufficient positive reinforcement **or** trusted human verification.

### **Open-Source Engineers**

Builders who value transparency, composability, and zero vendor lock-in.

* **Why it matters**
  Everything is inspectable: deterministic mappings (`explicit_terms`, `explicit_terms_text`, provenance fields), an external NER service, and a clear dual-store plan. BM25 queries are auditable and vector retrieval is constrained by BM25 anchors to avoid semantic drift.

* **Extensible data modeling**
  Swap in domain NER without changing the retrieval contract. Keep shared schemas stable while evolving analyzers and fielded clauses. Observability flags (e.g., saved JSONL) make experiments repeatable.

* **Use case**
  A programming Q&A assistant ingests manuals, API docs, and forum answers into **LT** with NER enrichments. At question time, NER extracts entities; the orchestrator **queries LT and HOT in parallel**, retrieves BM25 grounding, and adds vector context filtered to the anchored documents. A nightly job **materializes** popular LT docs into **HOT** via `/_reindex`; TTL cleans them up. No per-question copying, no mystery scores.

### **Enterprise Architects**

Leaders who must integrate AI into existing estates with guardrails for scale, security, and compliance.

* **Why it matters**
  Deterministic mappings, **entity-aware BM25 queries**, and **parallel LT+HOT search** keep behavior predictable, while vector augmentation improves semantic coverage without sacrificing governance.

* **Governance & compliance**
  Answers trace to documents and fields. Provenance (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) plus optional `hot_promoted_at` on materialized items support audits. **HOT → LT promotion** is gated by reinforcement or human sign-off.

* **Scalable infrastructure**
  Run **LT** for durability and **HOT** for volatile/experimental workloads with different SLAs. Capacity planning follows standard OpenSearch practices; snapshots on LT provide point-in-time attestations.

* **Use case**
  A policy assistant for a financial firm ingests manuals and memos into **LT**. Queries hit **LT and HOT** in parallel; BM25 provides the evidence and vector retrieval adds semantic context. Everything runs in-VPC with enterprise auth.

### **NetApp Infrastructure Customers**

Teams standardizing on ONTAP who want storage-level reliability with application-level transparency.

* **Why it matters**
  The architecture cleanly maps to storage controls: **LT** on durable tiers with snapshots/replication, **HOT** on low-latency media with low replication and TTL. The app path stays simple and auditable.

* **Performance on NetApp**
  Put **HOT** on faster storage to reduce tail latencies for BM25 queries; use caching (e.g., FlexCache) to keep hot index segments close to compute.

* **Use case**
  A support assistant indexes product manuals and KBs into **LT**. Queries run against **LT and HOT** in parallel and use vector context for semantic coverage. SnapMirror feeds a DR site and dev/test.

### **Cross-Industry Applicability**

The pattern stays the same; only the corpus and NER/BM25/vector implementation changes.

* **Healthcare**
  Clinical guidelines and formularies live in **LT** with provenance. Queries run against **LT and HOT**; BM25 grounding keeps evidence explicit, and vector context helps with paraphrased clinical phrasing. On-prem deployments align with HIPAA controls.

* **Retail & E-commerce**
  Product specs, return policies, and compatibility matrices sit in **LT**. Entity-aware BM25 queries retrieve precise answers; popular seasonal content can be materialized into **HOT** on a schedule and evicted by TTL, while vector context supports fuzzy matching.

* **Legal & Regulatory**
  Statutes, rulings, and memos are ingested into **LT** with stable IDs. Fielded, explainable matches power counsel workflows; any HOT presence is operational (materialized windows), not a hidden data path.

Across audiences, the benefits are consistent: **explainability, determinism, and operational control**. The dual-store design proves where answers came from, separates durable truth from volatile experiments, and turns policy enforcement into configuration—while keeping the request path fast and observable.

## **7. Implementation Guide**

Please see [GitHub repository](https://github.com/davidvonthenen/docuemnt-rag-guide) for:

- [Community / Open Source Implementation](./OSS_Community_Implementation.md)
 For a reference implementation, please check out the following: [community_implementation/README.md](./community_implementation/README.md)

- [Enterprise Implementation](./Enterprise_Implementation.md)
 For a reference implementation, please check out the following: [enterprise_implementation/README.md](./enterprise_implementation/README.md)

## **8. Conclusion**

The Hybrid RAG architecture moves AI retrieval from opaque heuristics to **observable, governable search**. Pairing a Large Language Model with **BM25-grounded retrieval** and **vector semantic augmentation** blends generation with verifiable evidence. Queries run against **both Long-Term (LT) and HOT (unstable)** in parallel, BM25 evidence is gathered from both stores, and vector context is added in a controlled, anchored way—maintaining reliability, transparency, and compliance that end-to-end training or vector-only stacks can’t match.

Knowledge is treated as a first-class asset. **LT** is the vetted, durable store with deterministic mappings (`explicit_terms`, provenance fields, versions). **HOT (unstable)** is an operational, entity-scoped store governed by TTL. **HOT → LT promotion happens only** when there is sufficient **positive reinforcement** of the data **or** a **trusted human-in-the-loop** has verified it.

Transparency is built in. Answers are grounded in retrievable documents with **explicit evidence blocks** and auditable query branches. The orchestrator’s observability controls (per-store totals, kept filepaths, and optional compact JSONL records) make the path from **question → entities → per-store results → answer** explainable and reproducible for reviewers and auditors.

Finally, this architecture aligns with enterprise governance. Benchmarks show **latency isn’t the primary reason** to split stores; two OpenSearch instances exist for **governance boundaries, retention variations control, and policy asymmetry**. Using **deterministic analyzers, explicit metadata, and TTL eviction**, the system meets accountability and regulatory needs without slowing delivery. Built on mature, open-source tools (OpenSearch, Flask, Python), it’s practical, scalable, and cost-effective. Hybrid RAG proves powerful AI can be both **capable and accountable**.
