# Enterprise Deployment: Hybrid RAG with NetApp Storage Integrations

This document outlines the **infrastructure deployment strategy** for the Hybrid RAG system. It assumes the use of the application logic found in the [Community Version](../community_version/README.md) but replaces the standard local storage with NetApp enterprise data services.

By decoupling the compute (OpenSearch/LLM) from the storage layer, we achieve strict Service Level Agreements (SLAs), distinct hardware QoS for "HOT" vs. "Long-Term" memory, and automated compliance governance—without changing the core application code.

## Architecture: The Storage Overlay

While the application logically sees two OpenSearch endpoints, the physical storage layer is architected to meet specific access patterns using NetApp ONTAP technologies.

| Logical Tier | Application Role | Physical Storage Configuration | NetApp Technology |
| --- | --- | --- | --- |
| **Long-Term Memory** | Authoritative Source of Truth; Read-Heavy; Compliance Required. | **Performance:** High-throughput NFS.<br><br>**Cost:** Auto-tiering enabled.<br><br>**Protection:** Zero-RPO replication. | **MetroCluster** (HA)<br><br>**FabricPool** (Tiering)<br><br>**SnapLock** (WORM) |
| **HOT Memory** | Unstable/Working Set; Mixed Read-Write; Low Latency. | **Performance:** In-memory speed with WAN caching.<br><br>**Isolation:** IOPS guarantees. | **FlexCache**<br><br>**Storage QoS**<br><br>**FlexClone** |

## 1. Long-Term Memory Configuration (Authoritative Store)

The Long-Term memory contains the vetted Knowledge Base. As this dataset grows to petabyte scale, keeping it entirely on high-performance SSDs is cost-prohibitive, yet it must remain instantly accessible for retrievals.

### Automated Cost Optimization (Auto-tiering)

We utilize **NetApp Auto-tiering** for the OpenSearch data shards backing the Long-Term instance.

* **Cold Data Movement:** Data blocks that are not accessed for a configurable period (default: 31 days) are automatically moved to a lower-cost object storage tier (S3).

* **Transparent Retrieval:** The application remains unaware of data movement; if the RAG agent retrieves an old fact, it is seamlessly recalled to the hot tier.

* **Intelligent Caching:** The storage detects large sequential reads (common during OpenSearch segment merging or backups) and bypasses the hot tier, preventing maintenance tasks from evicting active knowledge.

### Ingestion Pipeline (NetApp XCP)

For the initial population of Long-Term memory from legacy data lakes (Hadoop/HDFS or massive NFS shares), standard copy tools are insufficient.

* **High-Throughput Migration:** We utilize **NetApp XCP** to move data from legacy HDFS/NFS silos into the RAG source volume.

* **Verification:** XCP performs file verification to ensure the "Source of Truth" in the RAG system matches the source bit-for-bit, a critical requirement for regulated industries.

## 2. HOT Memory Configuration (The Working Set)

The HOT memory handles user-specific context, new unverified facts, and high-frequency queries. It requires extremely low latency and isolation from the heavy I/O of the Long-Term ingest processes.

### Global Locality (FlexCache)

In distributed inference clusters (e.g., Kubernetes nodes across zones), we back the HOT OpenSearch instance with a **NetApp FlexCache** volume.

* **Burst Performance:** Provides a sparse, writable cache that brings the active working set close to the GPU inference compute.
* **Reduce Latency:** Frequently accessed embeddings and indices are cached locally, ensuring sub-millisecond retrieval times for the "Unstable" memory tier.

### Noisy Neighbor Isolation (Storage QoS)

RAG ingestion jobs (writing to Long-Term) can saturate bandwidth. To protect the user experience on the HOT tier:

* **Min/Max IOPS:** We apply **Storage QoS** policies to the HOT volume to guarantee minimum throughput for user queries, ensuring that background ingest jobs never starve the inference engine.

### Migration & Re-indexing (Hot Tier Bypass)

When re-indexing data or migrating the HOT instance, we enable **Hot Tier Bypass**. This forces write operations directly to the cold tier, preventing bulk administrative operations from flooding the FlexCache and evicting the actual user context.

## 3. Data Protection and Governance

### Zero Data Loss (MetroCluster)

For the Long-Term memory, we utilize **NetApp MetroCluster**. This provides synchronous replication, ensuring that even in the event of a total site failure, the authoritative Knowledge Base is preserved with **Zero RPO**, allowing the RAG agent to resume operations immediately.

### Compliance Snapshots (SnapCenter & SnapLock)

* **SnapCenter:** Used to take application-consistent snapshots of the OpenSearch indices. This creates an immutable audit trail of exactly what the AI "knew" at any specific point in time.
* **SnapLock:** For strict regulatory environments, SnapLock enforces WORM (Write Once, Read Many) protection on the Long-Term volume, mathematically proving that the source data has not been tampered with.

## 4. Deployment Configuration

The following Docker configuration demonstrates how to map the OpenSearch containers to the specific NetApp volumes described above. This replaces the standard local volume mapping from the Community Version.

```bash
# create a docker network
docker network create opensearch-net

# ---------------------------------------------------------
# LONG-TERM MEMORY (Authoritative)
# Backed by NetApp Volume with Auto-tiering & MetroCluster
# ---------------------------------------------------------
docker run -d \
 --name opensearch-longterm \
    --network opensearch-net \
 -p 9201:9200 -p 9601:9600 \
    -e "discovery.type=single-node" \
 -e "DISABLE_SECURITY_PLUGIN=true" \
    # Map data to the NFS mount point protected by MetroCluster
    -v "/mnt/netapp_longterm_vol/data:/usr/share/opensearch/data" \
    # Map snapshots to a SnapLock compliance volume
    -v "/mnt/netapp_snaplock_vol/snapshots:/mnt/snapshots" \
 opensearchproject/opensearch:3.2.0

# Dashboards (Standard configuration)
docker run -d \
 --name opensearch-longterm-dashboards \
    --network opensearch-net \
 -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-longterm:9200"]' \
 -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0

# ---------------------------------------------------------
# HOT MEMORY (Working Set)
# Backed by FlexCache for high-speed local bursting
# ---------------------------------------------------------
docker run -d \
 --name opensearch-shortterm \
    --network opensearch-net \
 -p 9202:9200 -p 9602:9600 \
    -e "discovery.type=single-node" \
 -e "DISABLE_SECURITY_PLUGIN=true" \
    # Map data to the FlexCache volume for low-latency access
    -v "/mnt/netapp_flexcache_hot/data:/usr/share/opensearch/data" \
    # FlexClone is used here for instant test/dev copies
    -v "/mnt/netapp_flexcache_hot/snapshots:/mnt/snapshots" \
 opensearchproject/opensearch:3.2.0

# Dashboards (Standard configuration)
docker run -d \
 --name opensearch-shortterm-dashboards \
    --network opensearch-net \
 -p 5602:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-shortterm:9200"]' \
 -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0

```
