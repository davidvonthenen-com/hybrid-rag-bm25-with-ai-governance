# Hybrid RAG Using BM25: Better Option for AI Governance

Welcome to the **Hybrid RAG Guide**: a dual-memory (LT + HOT) Vector + BM25 approach using OpenSearch to Retrieval-Augmented Generation that delivers answers that are **transparent, deterministic, and governance-ready** by design. The repo houses two fully working paths:

* **Community / Open-Source Version**: a self-contained demo you can run on a laptop or in a pure open-source deployment.
* **Enterprise Version**: a production-grade variant that layers in ingest pipelines, NetApp FlexCache, SnapMirror, and other operational muscle.

![Hybrid RAG with Reinforcement Learning](./images/rag-hybrid-bm25.png)

By storing knowledge as **documents with enriched metadata** (named entities and provenance tags) instead of opaque vectors alone, the agent gains traceability, reduces hallucinations, and meets demanding audit requirements.

### Project Purpose

This project was developed to address limitations of traditional vector-centric RAG and to make retrieval **reproducible**, **explainable** and **auditable**.

* **Performance & latency**: The LT/HOT split exists primarily for **governance boundaries**, **retention variations control**, and **policy asymmetry**; retrieval remains lexical-first and observable.
* **Transparency and Explainability**: Vector embeddings are opaque. Document-based RAG stores explicit entity metadata (`explicit_terms`, `explicit_terms_text`) and uses fielded, auditable BM25 queries so you can show *why* a document matched.

Key objectives include:

* Provide a reference architecture for **Hybrid RAG** with explicit **HOT (unstable)** and **Long-Term (LT)** tiers.
* Make promotion from **HOT → LT** a **controlled event** that happens only when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified it.
* Show upgrade paths-from a minimal Python demo to an enterprise pipeline with NetApp enterprise storage.

## Benefits Over Graph-Based Hybrid RAG

While graph-based RAG is often touted for its deep relational mapping, it introduces significant operational friction and technical debt that a lexical-first (BM25 + Vector) Hybrid RAG architecture avoids. By grounding retrieval in explicit term matching and semantic similarity, enterprises achieve superior results with far less complexity:

* **Unified Infrastructure Stack:** Hybrid RAG utilizes a single software ecosystem (OpenSearch) for both BM25 lexical search and vector embeddings, whereas graph-based RAG typically requires a fragmented stack involving a dedicated graph database (e.g., Neo4j) alongside a separate vector store.
* **Elimination of Ontology Overhead:** Graph RAG requires the upfront creation and continuous maintenance of complex ontologies and schemas; in contrast, Hybrid RAG uses deterministic analyzers and entity extraction that adapt to new data without manual relational mapping.
* **Lower Technical Barrier to Entry:** Implementing BM25 does not require specialized graph theory expertise or "Graph Data Scientists," allowing existing software engineering and DevOps teams to manage the system using standard search engine patterns.
* **Deterministic Fact Traceability:** Lexical search provides immediate, human-readable evidence of why a document was retrieved through explicit keyword highlighting and field matching; a "glass box" approach compared to the often opaque traversal logic of multi-hop graph queries.
* **Operational Resilience at Scale:** Managing a graph at petabyte scale often leads to performance bottlenecks during high-concurrency writes; Hybrid RAG leverages mature enterprise storage features like FlexCache and MetroCluster to scale retrieval and protection without the fragility of massive interconnected nodes.

## Benefits Over Vector-Based RAG

While traditional vector-only RAG excels at semantic "vibes," it often struggles with the precision, auditability, and deterministic grounding required by the enterprise. Transitioning to a lexical-first (BM25 + Vector) architecture provides several critical advantages:

* **Deterministic Factual Grounding:** BM25 eliminates the "black box" of vector-only ranking by providing traceable, keyword-based evidence for every retrieval, ensuring exact matches for entities like product IDs or legal clauses.
* **Audit-Ready Provenance:** By utilizing explicit metadata fields (e.g., doc_version, ingested_at_ms), every fact used to ground an LLM response is tied to a verifiable source of truth, satisfying regulatory requirements for data lineage.
* **Mitigation of Semantic Drift:** Unlike vector embeddings, which can suffer from "hallucinated similarity" in high-dimensional space, lexical search anchors the retrieval in explicit term frequency, preventing contextually irrelevant documents from polluting the LLM prompt.
* **Storage-Aware Compliance:** Integrating with NetApp storage allows for immutable SnapCenter snapshots and MetroCluster replication, ensuring that the retrieved context is protected by enterprise-grade disaster recovery and point-in-time audit capabilities.
* **Operational Explainability:** Use of deterministic analyzers and keyword highlights makes it straightforward for human reviewers to understand exactly why the RAG agent selected a specific snippet, reducing the risk of hidden bias in embedding-only systems.

## Data Relationships

![Comparison: Retrieval in RAG Explained](./images/data-points-bm25.png)

The following analysis breaks down the data relationships and operational characteristics of the three primary retrieval methodologies used in modern RAG systems:

### Vector Embeddings (Approximate Nearest Neighbor)

* **Isolated Data Points:** In this model, data exists as high-dimensional points in a vector space with no explicit, hard-coded relationships between them.
* **Semantic Proximity:** Relationships are purely mathematical and based on "semantic similarity". Retrieval is determined by the distance between a query vector and document vectors.
* **Limitations:** Because there are no formal links, this approach often suffers from "semantic drift" or "hallucinated similarity," where the system retrieves contextually irrelevant information that happens to be mathematically nearby.

### Knowledge Graph

* **Explicit Relationships:** This represents the gold standard for relational mapping, where every entity is a node connected by explicit edges (relationships) to other nodes.
* **Deep Contextual Traversal:** It allows for complex, multi-hop reasoning by following defined paths between data points.
* **High Operational Burden:** While ideal for precision, graphs are notoriously difficult to manage at enterprise scale, requiring specialized expertise, complex ontology creation, and significant development costs to keep relations updated over time.

### BM25 (Lexical-First Hybrid)

* **The "Middle Ground" Infrastructure:** As shown in the image, BM25 provides structured data relationships without the overhead of a formal graph.
* **Deterministic Keyword Matching:** It uses deterministic lexical logic to anchor retrieval in specific term frequencies and exact keyword matches.
* **Efficient Logic Layer:** By using entity enrichment (like the external NER service used in our Hybrid RAG pipeline), we can create simple, auditable relationships between terms and documents.
* **Optimized for Enterprise Storage:** Unlike fragmented graph databases, BM25 utilizes a unified software stack (OpenSearch) that integrates seamlessly with NetApp features like FlexCache for micro-second local reads and Auto-tiering to move cold shards to S3 automatically.

## Community vs Enterprise: What Changes?

| Capability                 | Community Edition                             | Enterprise Edition                                                          |
| -------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **HOT backing store**      | `tmpfs` / RAM-disk on the dev box             | **NetApp FlexCache** for locality; (Optional) **SnapMirror** replicas       |
| **LT backing store**       | Rotating Disk, SSD, etc.                      | **NetApp FlexCache** for locality; Durability via **SnapMirror** replicas   |
| **Governance hooks**       | Basic provenance tags                         | Same as Community Version                                                   |
| **Latency posture**        | Latency is secondary to governance boundaries | Latency tuned per SLA, but the split is for **governance/policy asymmetry** |

> **TL;DR:** start with the community guide for laptops and commodity hardware; switch to the enterprise path when you need multi-site, 24×7, and **governance** at scale. The split exists for **control and auditability**, not because latency forces it.

## Where To Dive Deeper

| Hybrid RAG                                   | What it covers                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Hybrid Search for Better AI Governance**   | Vision & governance rationale for Hybrid RAG [link](./Hybrid_Search_for_Better_AI_Governance.md)  |
| **Community Version Guide**                  | Step-by-step setup for the open-source flavour [link](./OSS_Community_Version.md)                     |
| **Community README**                         | Hands-on commands & scripts [link](./community_version/README.md)                                     |
| **Enterprise Version Guide**                 | Deep dive on ingest pipelines, FlexCache, SnapMirror [link](./Enterprise_Version.md)                  |
| **Enterprise README**                        | Production deployment notes [link](./enterprise_version/README.md)                                    |

## Quick Start

```bash
# 1. Clone the repo
$ git clone https://github.com/NetApp/hybrid-rag-bm25-with-ai-governance.git

# 2. Pick your path
$ cd dhybrid-rag-bm25-with-ai-governance/community_version  # laptop demo or open source deployment
# or
$ cd hybrid-rag-bm25-with-ai-governance/enterprise_version  # production-ready deployment

# 3. Follow the README in that folder
```

Questions? Open an issue or start a discussion and contributions are welcome!
