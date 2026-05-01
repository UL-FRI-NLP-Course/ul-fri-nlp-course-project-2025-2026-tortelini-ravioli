import time
import json
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Set up models (same as query.py)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)
Settings.llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=120.0,
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("grafana")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",
)

test_queries = [
    "What is Grafana used for?",
    "How do I create a new dashboard?",
    "Explain how to set up alerts.",
    "What data sources does Grafana support?",
    "How can I share a dashboard with others?",
    "Describe the process to install plugins.",
    "How do I manage user permissions?",
    "What is the default port for Grafana?",
    "How to upgrade Grafana to the latest version?",
    "Where are Grafana logs stored?",
    "How do I change the Grafana admin password?",
    "How do I add a Prometheus data source?",
    "Can Grafana send alert notifications?",
    "How do I back up Grafana?",
    "Where is the Grafana configuration file located?",
    "How do I invite another user to Grafana?",
    "How do I install a panel plugin?",
    "Can Grafana connect to Elasticsearch?",
    "How do I export a dashboard?",
    "What happens if Grafana cannot find a data source?"
]

results = []

for query in test_queries:
    start = time.time()
    response = query_engine.query(query)
    end = time.time()
    response_time = end - start
    sources = [n.metadata.get("file_path", "unknown") for n in response.source_nodes]
    results.append({
        "query": query,
        "response_time_sec": response_time,
        "response": str(response),
        "sources": sources
    })
    print(f"Query: {query}\nTime: {response_time:.2f}s\nSources: {sources}\n---\n")

with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Benchmark complete. Results saved to benchmark_results.json.")
