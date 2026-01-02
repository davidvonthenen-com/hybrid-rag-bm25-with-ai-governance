# Hybrid RAG Using BM25: Better Option for AI Governance

Welcome to the **Hybrid RAG Guide**: a dual-memory (LT + HOT) Vector + BM25 approach to Retrieval-Augmented Generation that delivers answers that are **transparent, deterministic, and governance-ready** by design. The repo houses two fully working paths:

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

## Benefits Over Vector-Based RAG

Adopting Hybrid RAG addresses several key limitations of traditional RAG agents:

* **Multi-Step Reasoning**: OpenSearch indexing with entity enrichment supports complex, entity-aware queries across related documents.
* **Bias Mitigation**: Explicit metadata and annotations expose skew and make it easier to correct.
* **Improved Compliance and Governance**: Provenance fields and auditable clauses enable traceable, regulator-friendly retrieval.
* **Risk Management**: Deterministic lexical logic and highlights reduce hallucinations and make review straightforward.

### Community vs Enterprise: What Changes?

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
