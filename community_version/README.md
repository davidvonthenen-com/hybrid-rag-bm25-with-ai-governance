# Pure Open Source Community Implementation for Document RAG with Reinforcement Learning

This README.md provides an **open-source/community-oriented reference implementation** for a  lexical-first Retrieval-Augmented Generation (RAG) system, which supports a HOT data and reinforcement learning to consume new facts. This is modeled after a specific example, so your implementation might differ materially from the specifics here; however, the high-level concepts are the same.

## Prerequisites

- A Linux or Mac-based development machine with sufficient memory to run two OpenSearch instances and an LLM (≈8B parameters).
  - *Windows users:* use a Linux VM or cloud instance if possible.
- **Python 3.10+** installed (with [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for isolation).
- **Docker** installed (for running OpenSearch, etc.).
- Basic familiarity with shell and Docker commands.

**Docker images to pre-pull:**

- `opensearchproject/opensearch:3.2.0` (used for both long-term and HOT instances)

### LLM to pre-download:

For example, you can use the following 7-8B parameter models that run locally (CPU-friendly via [llama.cpp](https://github.com/ggerganov/llama.cpp)):

- Intel's **neural-chat-7B-v3-3-GGUF** - *(tested model)* available on HuggingFace
- OR [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) - an 8B instruction-tuned Llama variant.
- OR use an API/manager like [Ollama](https://ollama.com/), e.g. `ollama pull llama3:8b` for an 8B Llama model.

## Setting Up the Environment

To get started, we need to set up two main components of our environment: a OpenSearch database and a local LLM for question-answering. We'll use **Docker** to run OpenSearch. You'll need to download an LLM (we'll provide some recommendations) and set up a Python environment for our code.

### Demonstration Purposes

For demonstration purposes, we will create 2 OpenSearch instances: one instance for Long-Term vetted data and second instance for Hot / unstable data.

### Launch OpenSearch with Docker

Create a private network and bring up **long-term** and **HOT** instances. We’ll expose **9201** (LT) and **9202** (HOT) on localhost and disable the security plugin for a frictionless demo.

```bash
# create a docker network so that opensearch instances and dashboards can access each other
docker network create opensearch-net

# opensearch for long-term data retention
docker run -d \
    --name opensearch-longterm \
    --network opensearch-net \
    -p 9201:9200 -p 9601:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -v "$HOME/opensearch-longterm/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch-longterm/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.2.0

# (optional) opensearch dashboards for debugging, observability
docker run -d \
    --name opensearch-longterm-dashboards \
    --network opensearch-net \
    -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-longterm:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0

# opensearch for HOT data
docker run -d \
    --name opensearch-shortterm \
    --network opensearch-net \
    -p 9202:9200 -p 9602:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -v "$HOME/opensearch-shortterm/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch-shortterm/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.2.0

# (optional) opensearch dashboards for debugging, observability
docker run -d \
    --name opensearch-shortterm-dashboards \
    --network opensearch-net \
    -p 5602:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-shortterm:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0
```

This will download the OpenSearch image (if not already present) and start two OpenSearch servers in the background. The OpenSearch database will be empty initially. You can verify it's running by opening the OpenSearch Browser at **[http://127.0.0.1:9201](http://127.0.0.1:9201)** and **["http://127.0.0.1:9202]("http://127.0.0.1:9202)** in your browser. 

> **IMPORTANT:** The password for the `Long-term Memory` and the `Short-term Memory` instance has been disable for ease of use. In production environments, remove this line `DISABLE_SECURITY_DASHBOARDS_PLUGIN=true` from the docker command for starting up the containers.

### Python Environment and Dependencies

With the OpenSearch instances running and the model file ready, set up a Python environment for running the provided code. You should have Python 3.10+ available. It's recommended to use a virtual environment or a Conda environment for the lab.

Install the required Python libraries using pip. A convenient `requirements.txt` file has been provided for you. 

```bash
pip install -r requirements.txt
```

After installing spaCy, download the small English model for NER:

```bash
python -m spacy download en_core_web_sm
```

### Start the NER HTTP Service

This solution uses a default Named Entity Recognition model for association `keywords` to `documents`. It's highly recommended that if your problem or domain area uses unique language set (for example, medical, legal, etc), the keyword selection will highly benefit from using a language specific NER model. 

To launch the packaged NER endpoint (default `http://127.0.0.1:8000/ner`) to extract entities, run the following command on a bash terminal:

```bash
python ner_service.py
```

## Background on the Data

Our knowledge source is a collection of BBC news articles in text format, which can be found in the zip file [bbc-example.zip](./bbc-example.zip). This zip file contains a subset of 300 BBC news articles from the 2225 articles in the [BBC Full Text Document Classification](https://bit.ly/4hBKNjp) dataset. After unzipping the archive, the directory structure will look like:

```
bbc/
├── tech/
    ├── 001.txt
    ├── 002.txt
    ├── 003.txt
    ├── 004.txt
    ├── 005.txt
    └── ...
```

Each file is a news article relating to technology in the world today.

You may need to unzip the `bbc-example.zip` file, which you can do by running this script:

```bash
unzip bbc-example.zip
```

## Paragraph-Level Chunking Demo

The reference ingest (`community_version/ingest.py`) now writes **two** indices on every run:

* `INDEX_NAME` (default `bbc`) contains the full documents with metadata for provenance.
* `CHUNK_INDEX_NAME` (default `bbc-chunks`) stores deterministic paragraph slices. Each record carries `parent_filepath`, `chunk_index`, and `chunk_count` so you can trace the snippet back to the source file when auditors ask where a fact originated.

Query helpers in `community_version/common.py` point to the chunk index by default, so BM25 hits already map to paragraph-scale spans. Adjust `split_into_paragraphs()` if your corpus needs different chunking heuristics, or swap `LONG_INDEX_NAME`/`HOT_INDEX_NAME` env vars if you prefer document-level retrieval.

## Example Workflows

Here are different workflows that demonstrate how a Document RAG solution can lead to better AI Governance.

### 1. Simple Query Example

In this scenario, we will ingest data into the long-term OpenSearch and then execute a query to see the basic mechanics of retrieving data from OpenSearch..

1. **Perform the Ingest**: Run `python ingest.py`
   **WARNING:** This will erase any existing nodes and edges in your OpenSearch database and then reload the BBC dataset afresh. The ingest process will parse the documents, split them into paragraphs (the default chunking mode), perform NER to extract entities, and merge everything into the **long-term** OpenSearch instance and its companion chunk index. Each relationship is annotated with metadata, including its source document, ingestion timestamp, and a schema version tag for governance purposes.

2. **Perform a Simple Query**: Run `python query.py`
 This script poses a sample question to the system. This will retrieve facts from both the **long-term** and the **hot** OpenSearch instances. If you enable the debugging and tracing, you will see the query operation search for keywords in both OpenSearch instances. Since only the **long-term** instance has data, you will only see results from the **long-term** instance return back to the query.
  - To use Fireworks AI instead of the local OpenAI-compatible LLM server, set `FIREWORKS_API_KEY` and run `python query.py --fireworksai`. You can also set `FIREWORKS_AI=true` in your environment to make it the default.

### 2. Reinforcement Learning

This workflow illustrates how new facts can be introduced and evaluated in **hot** memory, and how the system utilizes a reinforcement learning (RL)-style feedback loop to determine which facts are promoted to long-term memory. We will simulate adding new information and then "teaching" the system through usage and validation. Not depicted in this example is the `CRON` job implementation, which invokes the following scripts: `expire_hot_data.py` and `python manual_promote.py` to handle promotion and expiration of facts from **hot** memory.

1. **Perform the Ingest**: Run `python ingest.py`
   **WARNING:** This will erase any existing nodes and edges in your OpenSearch databases (both **long-term** and **hot**) and then reload the BBC dataset afresh. The ingest process will parse the documents, split them into paragraphs (our default chunk size), perform NER to extract entities, and merge everything into the **long-term** OpenSearch instance alongside the paragraph chunk index. Each relationship is annotated with metadata, including its source document, ingestion timestamp, and a schema version tag for governance purposes.

2. **Enter New Facts Into HOT Memory**: Run `python example/reinforcement_learning.py`
 This example script will prompt you to introduce five new "facts" into the **hot** memory. These could be considered insights or data points not present in the original dataset. For example, facts #1 and #3 are about **OpenAI** (which are not in the BBC tech articles by default). You can choose to inject all or some of these facts; we recommend selecting **fact #1 and fact #3** for this demo. In this simplified script, the use of facts through the queries serves as a reinforcement signal.

3. **Perform a Simple Query**: Run `python example/query.py`
 This script poses a sample question based on the **OpenAI** recommended to add into **hot memory** (ie, **fact #1 and fact #3**). A specific question based on these **OpenAI** facts will be asked, this will retrieve facts from both the **long-term** and the **hot** OpenSearch instances. If you enable the debugging and tracing, you will see the query operation search for keywords in both OpenSearch instances.

4. **Mark One Fact Into Long-Term Memory**: Run `python manual_promote.py`
 This script allows us to flag specific **hot** facts as *validated* (worthy of long-term preservation) and others as *expired*. Following the recommendation, select the fact you added as #1 (e.g., the first OpenAI fact) to **promote**, and select fact #3 to let it *expire*.

5. **Expire HOT Memory/Cache**: Run `python expire_hot_data.py`
 This will remove any remaining facts in the **hot** instance that have expired or were marked to expire (in our case, fact #3, which we did not promote). This step simulates the cache eviction process for facts that prove not to be useful. In a production environment, such expiration could also be handled by Kafka Connect emitting a *tombstone* event or by a scheduled job that prunes expired data. (For instance, if a fact is not promoted and its time-to-live lapses, it will simply disappear from the cache, leaving only long-term facts stored permanently.)

6. **Verify Fact Can Be Queried**: Run `python example/query.py`
 This will query both OpenSearch instances and if you have the logging/debugging enabled in the script, you will notice that the **OpenAI** fact is now retrieved from the **long-term** instance. This demonstrates that the fact was successfully carried over. In contrast, if you query for fact #3 (the one we let expire), it will not be found in either database (having been purged from **hot** and never added to **long-term**).

7. **Reset Both Long-Term and HOT Instances**: Run `python helper/wipe_all_memory.py`.
 This stops and/or clears the data in both the long-term and **hot** databases, allowing you to repeat the workflows from a clean state if desired.

## External BM25 Document Re-Ranker

All query entry points (`query.py`, `example/query.py`, and the reinforcement-learning demo) call `common.ask(...)`, which now defaults to `external_ranker=True`. After OpenSearch returns results from **both** LT and HOT, the helper uses the lightweight [`bm25s`](https://github.com/xhluca/bm25s) library to re-rank the merged hit list locally. This keeps score math transparent, produces a single ordered context list for the LLM, and makes it obvious how each snippet earned its place. If you ever want to inspect raw OpenSearch ranking, pass `external_ranker=False` when calling `ask` or fork the script to expose a CLI flag.

## Conclusion

By following these workflows, you have deployed a dual-memory Document RAG Agent. We utilized Docker CLI commands to orchestrate a OpenSearch-based **long-term memory** and **hot memory** to build out this solution. The result is a retrieval-augmented generation pipeline that is **fast, transparent, and robust**: delivering sub-second query responses with full traceability of which facts were used and how they were processed through the system.

For further exploration, consider pointing the ingestion pipeline at your data sources. You can also integrate a larger language model or an API-based model for the LLM component if needed.

> **IMPORTANT:** This is modeled after a specific example, so your implementation might differ materially from the specifics here; however, the high-level concepts are the same.

We hope this reference implementation provides a solid foundation for building **production-ready, enterprise-scale Document RAG** solutions. By combining OpenSearch databases with Kafka-based CDC and advanced storage like FlexCache, you can achieve AI systems that are **faster, clearer, safer, and compliant by design**. Happy building!
