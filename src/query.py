import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

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

# REPL
print("Ready. Type 'quit' to exit.\n")
while True:
    question = input("Query> ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    response = query_engine.query(question)
    print(f"\n{response}\n")

    # Show which source files were used
    sources = {n.metadata.get("file_path", "unknown") for n in response.source_nodes}
    print("Sources:")
    for s in sorted(sources):
        print(f"  {s}")
    print()