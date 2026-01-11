# Vector-Databases-for-Semantic-Search 
##### *Study and presentation of a scientific article about Vector Databases for Semantic Search(T8)*

##### *BDNSV final project: FMI - University of Bucharest, Year III 2025-2026, Onisie Andreea*

## 1. Project Overview
_Main idea:_

This project presents an overview of vector databases and their role in enabling semantic search over unstructured data. It explains how embeddings allow systems to capture meaning and perform similarity search, supporting applications such as search engines and recommendation systems. The report also covers key indexing techniques and shows how modern vector databases balance accuracy, speed, and memory efficiency.

In addition, the project focuses on the integration of vector databases into AI pipelines, particularly in combination with large language models through retrieval-augmented generation (RAG). Recent trends such as semantic compression, graph-augmented retrieval, hybrid search, and multimodal embeddings are highlighted, showing how these approaches improve performance and support multiple data types.

Finally, future research directions are outlined, including self-updating indexes, hardware acceleration, and scalable architectures. Overall, vector databases are presented as a core component of modern AI systems, enabling efficient and meaningful use of large-scale unstructured data.

_Content:_

<img width="287" height="543" alt="image" src="https://github.com/user-attachments/assets/8cda2401-8bba-4406-a805-450f71a24406" />

---
Inspired by the ideas presented in the “Beyond Nearest Neighbors” article (https://arxiv.org/pdf/2507.19715), I also implemented a small-scale semantic search system using open-source tools. 

Text embeddings are generated locally and then compressed using dimensionality reduction techniques. The system performs semantic search both before and after compression, allowing a direct comparison between full and compressed embeddings. 

This experiment aims to show that compressed embeddings can still support meaningful semantic search while reducing vector size and improving efficiency.

_Video with demo for semantic compression available here:_

_Code for demo implementation available here:_


## 2. Research Background
- _Key concepts_
    - Vector Databases: Definition and Purpose, Advantages and Limitations
    - Embeddings and Vector Representations
    - Keyword Search
    - Semantic Search based on Similarity Search
    - Indexing Techniques - Distance-based, Partition-based, Graph-based and Compression-based approaches
    - Modern Vector Database Systems - such as Milvus, Weaviate, Pinecone, Qdrand, Chroma, Redis, Elasticsearch and OpenSearch
    - Hybrid search
    - Semantic compression (DEMO)
    - Graph-augmented retrieval
    - Multimodal embeddings
    - Retrieval-Augmented Generation - RAG
    - Future Directions - Updating indexes, Scaling, Hardware acceleration
    - Integration with AI / LLMs

- _Methods used in the research_
    - **Embedding generation** – The analyzed papers use machine learning models to transform text, images, or other unstructured data into high-dimensional vector representations (embeddings), which capture semantic meaning.
    - **Similarity search** – Semantic search is performed, rather than keyword search, by comparing embeddings using distance or similarity measures to find semantically related items.
    - **Approximate Nearest Neighbor (ANN) search** – Most approaches rely on ANN methods that improve speed while maintaining acceptable accuracy.
    - **Indexing techniques for vectors** – The research discusses different indexing strategies, including distance-based, partition-based, graph-based, and compression-based methods, to organize vectors and reduce the number of comparisons during search.
    - **Compression techniques** – Semantic compression is introduced to reduce memory usage and to speed up similarity search while preserving accurate results.
    - **Hybrid search** - It allows systems to balance different goals by combining different approaches of indexing.
    - **Graph-Augmented Retrieval** - Vectors are organized into a graph structure, where each vector is connected to a small number of its nearest neighbors, improving efficiency.
    - **Multimodal Embeddings use** -  Combining different types of information in the same vector space  improves user experience and relevance.
    - **Integration with AI models** – Vector databases are often used together with large language models (LLMs).
      
- Software/tools mentioned  
- Personal understanding in simple terms  

## 3. System Architecture
### 3.1 Workflow Diagram
### 3.2 Data Model
### 3.3 Software & Hardware Configuration

## 4. Dataset

## 5. Implementation / Code Snippets
+screenshots (soon..)

## 6. Analysis & Interpretation
semantic search vs keyword search

example queries and retrieved results

efficiency, scalability, limitations

## 7. How to run the project
## 8. Notes / Limitations
## 9. AI & Tool Acknowledgment
## 10. Bibliography
[1]  James Jie Pan, Jianguo Wang & Guoliang Li, Survey of Vector Database Management Systems, 2023, https://arxiv.org/pdf/2310.14021, Last accessed: January 2026

[2] Cuzzocrea Alfredo, Vector Databases for Modelling, Managing and Querying Big Scientific Data: Models, Issues, Paradigms, 2025, https://dl.acm.org/doi/pdf/10.1145/3733723.3742469, Last accessed: January 2026

[3] Rahul Raja & Arpita Vats, Beyond Nearest Neighbors: Semantic Compression and Graph-Augmented Retrieval for Enhanced Vector Search, 2025, https://arxiv.org/pdf/2507.19715, Last accessed: January 2026

[4] OpenAI, ChatGPT, https://chatgpt.com/, Last accessed: January 2026
