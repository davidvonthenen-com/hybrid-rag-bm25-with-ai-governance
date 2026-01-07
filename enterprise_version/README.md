# Enterprise Implementation for Hybrid RAG with Cache and Reinforcement Learning

This README.md provides an **enterprise-oriented reference implementation** for a Hybrid Retrieval-Augmented Generation (RAG) system. It builds upon the [open-source community version](./community_version/README.md) by adding production-grade features: **Kafka Connect** pipelines for automated data sync, a **dual-memory model** backed by high-speed caching ([FlexCache](https://www.netapp.com/data-storage/what-is-flex-cache/)) and durable storage ([SnapMirror](https://docs.netapp.com/us-en/ontap/concepts/snapmirror-disaster-recovery-data-transfer-concept.html) replication), and a **reinforcement learning loop** that promotes facts via Kafka-triggered events.

Hybrid RAG explicitly combines **BM25 lexical search** for deterministic precision with **vector embeddings** for semantic context, thereby mitigating the hallucinations of pure vector approaches and the rigidity of keyword-only search. The integration of Apache Kafka's change-data-capture (CDC) ensures seamless synchronization between **HOT (unstable)** and **Long-Term (authoritative)** memory stores. Meanwhile, NetApp [FlexCache](https://www.netapp.com/data-storage/what-is-flex-cache/) ensures the HOT tier resides close to the compute, and [SnapCenter](https://www.netapp.com/snapcenter/) provides immutable compliance snapshots. The result is a faster, more transparent, and governable RAG system ready for enterprise workloads.

## Prerequisites

* A Linux or Mac-based development machine with sufficient memory to run two OpenSearch instances and an LLM (≈8B parameters).
  * *Windows users:* Use a Linux VM or WSL2.
* **Python 3.10+** installed (with [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for isolation).
* **Docker** installed (for running OpenSearch, Dashboards, Kafka, etc.).
* **Apache Kafka** with Kafka Connect available (e.g., via Confluent Platform Docker images) for streaming data between memory tiers.
* (Optional) Access to a **NetApp ONTAP** environment.
  * **FlexCache** is recommended for the **HOT** tier to allow caching across zones/regions.
* **FabricPool/Auto-tiering** is recommended for the **Long-Term** tier to move cold shards to object storage automatically.
* Basic familiarity with shell and Docker commands.

**Docker images to pre-pull:**

* `opensearchproject/opensearch:3.2.0` (For both Long-Term and HOT instances)
* `opensearchproject/opensearch-dashboards:3.2.0` (Visualization)
* Kafka and Zookeeper images (e.g., `confluentinc/cp-kafka:7.4.0`)
* Kafka Connect image (e.g., `confluentinc/cp-kafka-connect:7.4.0` with the OpenSearch Sink Connector installed)

### LLM to pre-download:

For example, you can use the following 7-8B parameter models that run locally (CPU-friendly via [llama.cpp](https://github.com/ggerganov/llama.cpp)):

* Alibaba Cloud's **[Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF)** - *(tested model)* available on HuggingFace
* Intel's **[neural-chat-7B-v3-3-GGUF](https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF)** - available on HuggingFace

## Setting Up the Environment

We will set up two OpenSearch instances: one for **Long-Term Memory** (authoritative, strictly governed) and one for **HOT Memory** (unstable, high-speed cache). We also deploy Apache Kafka to handle the "Reinforcement Learning" promotion loop.

### Launch OpenSearch and Kafka with Docker

Below are example Docker CLI commands. They configure two OpenSearch nodes.

* **Long-Term:** Uses standard storage (or NetApp NFS).
* **HOT:** In a production setup, the volume mapped to `/usr/share/opensearch/data` would be backed by **NetApp FlexCache**. This allows the HOT tier to burst to high throughput or be cached in a different availability zone near the GPU inference cluster.

```bash
# create docker network
docker network create opensearch-net

# Long-Term Memory Instance (opensearch-long-term)
# API: http://localhost:9201
# Admin Panel via Dashboards: http://localhost:5601
# Port 9201
docker run -d \
    --name opensearch-longterm \
    --network opensearch-net \
    -p 9201:9200 -p 9601:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -v "$HOME/opensearch-longterm/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch-longterm/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.2.0

docker run -d \
    --name opensearch-longterm-dashboards \
    --network opensearch-net \
    -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-longterm:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0


# HOT Memory Instance (opensearch-shortterm)
# API: http://localhost:9202
# Admin Panel via Dashboards: http://localhost:5602
# Port 9202
# The volume below would ideally be a FlexCache volume for burst performance
docker run -d \
    --name opensearch-shortterm \
    --network opensearch-net \
    -p 9202:9200 -p 9602:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -v "$HOME/opensearch-shortterm/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch-shortterm/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.2.0

docker run -d \
    --name opensearch-shortterm-dashboards \
    --network opensearch-net \
    -p 5602:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-shortterm:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0
```

> 
> **NetApp Storage Note:** If implementing **Auto-tiering** on the Long-Term instance, data that hasn't been queried (read) for a configurable period (default 31 days) moves to a cold object tier (S3), reducing the TCO of your authoritative knowledge base. The `Auto-tiering` feature detects large sequential reads (like re-indexing) and allows them to bypass the hot tier to prevent "re-heating" archived data unnecessarily.
> 

Launch your Kafka services (Zookeeper, Broker, Connect) using your preferred Compose file or CLI commands. Ensure they share a network with the OpenSearch containers.

### Configuring Kafka Connectors

We use Kafka to decouple the governance logic from the database.

**OpenSearch Source (HOT → Kafka):** This connector (or a lightweight Python producer in this reference impl) watches the **HOT** index. When a document's "confidence score" crosses the promotion threshold (indicated by a `promoted: pending` tag), it publishes the document to a `promotions.topic`.

**OpenSearch Sink (Kafka → Long-Term):** This connector consumes the `promotions.topic` and indexes the verified document into the **Long-Term** OpenSearch instance. Crucially, this sink acts as a **WORM (Write Once, Read Many)** gateway—only validated facts enter here.

### Python Environment and Dependencies

Install the required libraries. This reference uses `opensearch-py` for database interaction and `spacy` for the NER (Named Entity Recognition) required by Hybrid RAG.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Background on the Data

Our example knowledge source is the **bbc-example.zip** (subset of 300 BBC news articles).
In an enterprise scenario, you might have petabytes of legacy data in Hadoop or HDFS. For such cases, we recommend using **NetApp XCP** to migrate that data into the NFS volume backing the **Long-Term** OpenSearch instance. XCP provides high-throughput migration and verification, ensuring your "Source of Truth" is bit-perfect before ingestion.

If you have not already, unzip the data:

```bash
unzip bbc-example.zip
```

## Example Workflows

### 1. Simple Query Example (Hybrid Retrieval)

In this scenario, we ingest data into the **Long-Term** memory and perform a retrieval that uses both BM25 (keyword) and Vector (semantic) search.

1. **Perform the Ingest**: Run `python ingest.py`.
This loads the BBC dataset into the **Long-Term** instance. It chunks documents, extracts entities (NER), and creates embeddings.
*Enterprise Note:* In production, you would use **SnapCenter** to take a snapshot of the volume immediately after a bulk ingest to establish a compliance baseline.
2. **Perform a Query**: Run `python hybrid_query.py`.
The script sends the user query to the **HOT** instance first.
* **Cache Miss:** If not found, it queries the **Long-Term** instance using Hybrid Search (BM25 + Vector).
* **Cache Populate:** It retrieves the relevant chunks and "hydrates" the **HOT** instance with them. This HOT instance uses a permissive schema for speed.
* **Result:** The LLM generates an answer using the retrieved context.

Future queries on this topic will hit the **HOT** instance (backed by **FlexCache**) first. FlexCache ensures that even if the Long-Term storage is in a different region, the cached working set is available locally with microsecond latency.

### 2. Reinforcement Learning Example

This workflow illustrates how the "Unstable" HOT tier allows for the introduction of new, unverified facts, and how the system uses usage frequency (RL) to decide what becomes permanent.

1. **Inject New Facts**: Run `python example/inject_unverified_fact.py`.
This script inserts a "rumor" or new data point (e.g., "Company X is acquiring Company Y") into the **HOT** index with `confidence_score = 1`.

2. **Simulate Usage**: Run `python example/simulate_usage.py`.
This script simulates users asking questions that rely on this new fact.
* Every time the fact is retrieved and the user provides positive feedback (simulated), the `confidence_score` increments.
* *Governance:* If the fact is flagged as "hallucination" by a user, the score decrements.

3. **Promotion Trigger**: Run `python example/trigger_promotion.py`.
Once the score hits a threshold (e.g., 25), the script tags the document as `status: promoted`.
* **Kafka Event:** This change is picked up by the pipeline and sent to the **Long-Term** sink.
* **Ingest:** The fact is permanently written to Long-Term memory.
* **Cleanup:** The system runs an eviction policy on the HOT tier.

*Storage Optimization:* If the **HOT** tier grows too large, the **Auto-tiering** logic can aggressively move cold, un-promoted data to cheaper object storage tiers, keeping the FlexCache volume lean and performant.

### 3. Eviction and Compliance

1. **Expire HOT Memory**: Run `python helper/evict_hot.py`.
This script removes documents from the HOT tier that have not been promoted and have exceeded their TTL (Time To Live).
  * *Note:* In the **Auto-tiering** context, we can set a policy where data unread for 31 days (default) is considered cold. This script mimics that logic application-side.

2. **Verify Long-Term**: Run `python example/verify_long_term.py`.
Query the authoritative index. You will see the promoted fact is now permanent. Un-promoted facts are gone.

## Conclusion

By following these workflows, you have deployed a **Dual-Memory Hybrid RAG** system.

* **Long-Term Memory:** Optimized for cost and scale using **NetApp Auto-tiering** (moving cold data to S3) and **XCP** for massive ingestion.
* **HOT Memory:** Optimized for speed using **FlexCache** to keep the active working set close to the LLM.
* **Governance:** The Kafka-based RL loop ensures that only validated data enters the permanent record, while **SnapMirror** ensures that record is replicated for disaster recovery.

This architecture solves the "Stale Data" problem of standard RAG and the "Hallucination" problem of pure vector databases, all while running on storage infrastructure that meets enterprise SLAs. Happy building!
