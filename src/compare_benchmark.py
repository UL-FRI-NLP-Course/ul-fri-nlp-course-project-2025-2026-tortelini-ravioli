import time
import json
import argparse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

parser = argparse.ArgumentParser(description="Compare Chroma RAG vs direct Ollama")
parser.add_argument("--chroma-path", default="./chroma_db", help="Path to existing Chroma DB")
parser.add_argument("--collection", default="grafana", help="Existing Chroma collection name")
parser.add_argument("--output", default="benchmark_compare_results.json", help="Output JSON path")
args = parser.parse_args()

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)
Settings.llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=120.0,
)

# Open existing Chroma DB/collection instead of creating one.
chroma_client = chromadb.PersistentClient(path=args.chroma_path)
collection = chroma_client.get_collection(args.collection)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",
)

direct_llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=120.0,
)

test_queries = [
    "What is Grafana used for?",
    "How do I create a new dashboard in Grafana?",
    "Explain how to set up alerts in Grafana.",
    "What data sources does Grafana support?",
    "How can I share a dashboard with others in Grafana?",
    "Describe the process to install plugins in Grafana.",
    "How do I manage user permissions in Grafana?",
    "What is the default port for Grafana?",
    "How to upgrade Grafana to the latest version?",
    "Where are Grafana logs stored?",
    "A team wants to migrate Grafana to a new server, keep existing dashboards, preserve user access, and avoid breaking plugins. What should they back up and reconfigure?",
    "An administrator upgraded Grafana and now alerts are not firing, some dashboards show missing data sources, and users report access issues. What should be checked first and in what order?",
    "How would you set up Grafana for a team where developers can edit dashboards, managers can only view them, and alert notifications must still reach both groups when needed?",
    "If a dashboard must be shared externally but the underlying data source and editing permissions should remain restricted, what Grafana features or settings are relevant?",
    "What steps would be involved in setting up a completely new Grafana instance with a Prometheus data source, alerting, user access control, and a shareable dashboard?",
    "What files and directories are important when troubleshooting Grafana startup problems, especially when checking logs and configuration?",
    "How do I troubleshoot a situation where alert rules exist but notifications are not being sent in Grafana?",
    "How can I configure Grafana so that only certain users can access a specific dashboard while others can still use the rest of the system?",
    "If I want to upgrade Grafana safely on a production system, what preparation and backup steps should I perform first?",
    "How do I install a plugin, verify that Grafana recognizes it correctly, and troubleshoot the issue if it does not appear in the UI?"
]

results = []

for query in test_queries:
    rag_start = time.time()
    rag_response = query_engine.query(query)
    rag_end = time.time()
    rag_sources = [n.metadata.get("file_path", "unknown") for n in rag_response.source_nodes]

    direct_prompt = f"Answer this question clearly and concisely. If you are unsure, say so.\n\n{query}"
    direct_start = time.time()
    direct_response = direct_llm.complete(direct_prompt)
    direct_end = time.time()

    results.append({
        "query": query,
        "rag": {
            "response_time_sec": round(rag_end - rag_start, 3),
            "response": str(rag_response),
            "sources": rag_sources,
        },
        "direct_ollama": {
            "response_time_sec": round(direct_end - direct_start, 3),
            "response": str(direct_response),
            "sources": [],
        }
    })

    print(
        f"Query: {query}\n"
        f"RAG Time: {rag_end - rag_start:.2f}s\n"
        f"RAG Sources: {rag_sources}\n"
        f"Direct Ollama Time: {direct_end - direct_start:.2f}s\n"
        f"---\n"
    )

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Benchmark complete. Results saved to {args.output}")