# Natural language processing course: `Grafana Documentation Chatbot`

## Proposed datasets:

- [Grafana Labs - Technical documentation](https://grafana.com/docs/)

## Future directions and ideas

One possible way to improve performance would be to clean the Grafana documentation by removing duplicates and organizing the source documents better. This can have a big effect on RAG systems, because cleaner and better-structured data usually leads to more accurate retrieval. We could also experiment with different chunking strategies to see whether any of them works significantly better than the others. For example, some strategies may preserve context better, while others may improve retrieval precision.

Another way to improve performance is to improve retrieval. This could be achieved by using a hybrid retrieval approach, which combines lexical and vector methods. Such an approach could help the system retrieve both exact keyword matches and semantically similar content. We could also try adapting retrieval to be query-specific, or additionally reordering the chunks after the initial retrieval to improve the quality of the final context given to the model.

The final idea for improvement is to try out different models and compare their performance. Some models might be better suited to this specific task than others, especially when dealing with technical documentation such as Grafana docs. In addition to answer quality, we could also compare the models based on speed, consistency, and hardware requirements.
