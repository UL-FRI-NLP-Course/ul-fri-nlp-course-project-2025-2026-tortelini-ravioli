# Natural language processing course: `Grafana Documentation Chatbot`

## Description

The goal of this project is to build an AI assistant that helps users access relevant parts of the Grafana documentation more easily. Grafana is a widely used open-source platform, but navigating its extensive documentation can sometimes be difficult. 

This chatbot uses a Retrieval-Augmented Generation (RAG) approach to retrieve the most relevant sections of the documentation based on a user's query. It then uses a language model to generate an accurate, context-aware answer grounded directly in the official documentation. By narrowing the focus specifically to official documentation, the assistant aims to deliver more precise and relevant answers, reducing the time needed to solve specific problems and learn new Grafana features.

## Proposed datasets:

- [Grafana Labs - Technical documentation](https://grafana.com/docs/)

## Installation

For detailed installation instructions and prerequisites (including Ollama and required Python packages), please see [src/DOCS.md](src/DOCS.md#building-the-rag-dataset).

## Usage

Once you have installed the required dependencies and cloned the documentation, you can run the following scripts from the `src` directory:

1. **Prepare the database:** Build the vector database using the downloaded documentation:
   ```bash
   cd src
   python prepare-db.py
   ```

2. **Query the documentation:** Start an interactive REPL session to query the Grafana documentation:
   ```bash
   python query.py
   ```

3. **Run benchmarks:** Evaluate the pipeline with the provided benchmark suite:
   ```bash
   python benchmark.py
   ```

## Future directions and ideas

One possible way to improve performance would be to clean the Grafana documentation by removing duplicates and organizing the source documents better. This can have a big effect on RAG systems, because cleaner and better-structured data usually leads to more accurate retrieval. We could also experiment with different chunking strategies to see whether any of them works significantly better than the others. For example, some strategies may preserve context better, while others may improve retrieval precision.

Another way to improve performance is to improve retrieval. This could be achieved by using a hybrid retrieval approach, which combines lexical and vector methods. Such an approach could help the system retrieve both exact keyword matches and semantically similar content. We could also try adapting retrieval to be query-specific, or additionally reordering the chunks after the initial retrieval to improve the quality of the final context given to the model.

The final idea for improvement is to try out different models and compare their performance. Some models might be better suited to this specific task than others, especially when dealing with technical documentation such as Grafana docs. In addition to answer quality, we could also compare the models based on speed, consistency, and hardware requirements.
