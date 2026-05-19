from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)
Settings.llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=600.0,
)

documents = SimpleDirectoryReader(
    input_dir="./grafana/docs/sources",
    recursive=True,
    required_exts=[".md"],
    filename_as_id=True,
).load_data()

pipeline = IngestionPipeline(
    transformations=[
        MarkdownElementNodeParser()
    ]
)

nodes = pipeline.run(documents=documents, num_workers=1)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("grafana")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
