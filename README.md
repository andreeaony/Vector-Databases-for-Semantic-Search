# Vector-Databases-for-Semantic-Search 
##### *Study and presentation of a scientific article about Vector Databases for Semantic Search(T8)*

##### *BDNSV final project: FMI - University of Bucharest, Year III 2025-2026, Onisie Andreea*

## 1. Project Overview
_Main idea:_

This project presents an overview of vector databases and their role in enabling semantic search over unstructured data. It explains how embeddings allow systems to capture meaning and perform similarity search, supporting applications such as search engines and recommendation systems. The report also covers key indexing techniques and shows how modern vector databases balance accuracy, speed, and memory efficiency.

In addition, the project focuses on the integration of vector databases into AI pipelines, particularly in combination with large language models through retrieval-augmented generation (RAG). Recent trends such as semantic compression, graph-augmented retrieval, hybrid search, and multimodal embeddings are highlighted, showing how these approaches improve performance and support multiple data types.

Finally, future research directions are outlined, including self-updating indexes, hardware acceleration, and scalable architectures. Overall, vector databases are presented as a core component of modern AI systems, enabling efficient and meaningful use of large-scale unstructured data.

_Content of research:_

<img width="287" height="543" alt="image" src="https://github.com/user-attachments/assets/8cda2401-8bba-4406-a805-450f71a24406" />

---
Inspired by the ideas presented in the [“Beyond Nearest Neighbors”](https://arxiv.org/pdf/2507.19715) article, I also implemented a small-scale semantic search system using open-source tools. 

Text data is loaded from a .csv file (AG News dataset from Kaggle). Sentence embeddings are generated using the pretrained all-MiniLM-L6-v2 model from HuggingFace (384 dimensions). Embeddings are then compressed using PCA to lower dimensions (64D and 32D) while preserving semantic information. Semantic search is performed on full embeddings and compressed embeddings for comparison.

This experiment aims to show that compressed embeddings can still support meaningful semantic search while reducing vector size and improving efficiency.

_Video with demo for semantic compression available here:_

_Code for demo implementation available here:_

_Notes:_
- Only the first 500 news descriptions are used for demonstration.
- PCA compression drastically reduces memory usage and speeds up search while keeping most semantic information intact.
- The plot provides a clear visual of the trade-offs between accuracy (recall), speed, and memory.

## 2. Project Background
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
      
- _Tools_
    - Python 3.12.3
    - Python Libraries:
        - pandas 2.3.3 – for reading and manipulating CSV datasets
        - numpy 2.4.1 – for numerical operations
        - matplotlib 3.10.8 – for plotting graphs and visualizations
        - scikit-learn 1.8.0 – for PCA (dimensionality reduction)
        - sentence-transformers 5.2.0 – for generating text embeddings
    - Kaggle – the source of the AG News dataset
    - HuggingFace – source of pretrained sentence embedding model all-MiniLM-L6-v2

## 3. System Architecture

The system is a semantic search demo showing how full embeddings vs compressed embeddings behave.

<img width="427" height="537" alt="image" src="https://github.com/user-attachments/assets/740e1d14-8109-4e24-8db4-5cffcfd813ea" />

### 3.1 Data Model

- Dataset: AG News dataset from Kaggle (first 500 documents used in demo), access it [here](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download)
- Columns used: Description (text)
- Embedding representation: 384D full embeddings, 64D and 32D compressed embeddings (model [all-MiniLM-L6-v2]( https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
  
### 3.2 Software & Hardware Configuration

_Software_
- **Python 3.12.3** – main programming language
- **pandas 2.3.3** – for reading and manipulating CSV datasets
- **numpy 2.4.1** – for numerical operations
- **matplotlib 3.10.8** – for plotting graphs and visualizations
- **scikit-learn 1.8.0** – for PCA (dimensionality reduction)
- **sentence-transformers 5.2.0** – for generating text embeddings
- **huggingface-hub** – for downloading pre-trained models from Hugging Face

_Hardware_
- **Processor**: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx, 2100 Mhz, 4 Core(s), 8 Logical Processor(s)
- **RAM**: 8 GB
- **OS**: Windows 11 Home
- **System Type**: x64-based PC

## 4. Implementation 

Loading data:
```
dataset = pd.read_csv("dataset.csv") 
```

Generating embeddings:
```
model = SentenceTransformer("all-MiniLM-L6-v2") #this is a pretrained open-source model with 384-dimensional embeddings from HuggingFace
fullEmbeddings = model.encode(documents, normalize_embeddings=True) #normalized for an easier cosine similarity calculation
```

Compressing embeddings - dimensionality reduction to 64D and 32D:
```
#64D
pca64 = PCA(n_components=64)
compressedEmbeddings64 = pca64.fit_transform(fullEmbeddings)
compressedEmbeddings64 /= np.linalg.norm(compressedEmbeddings64, axis=1, keepdims=True) #normalized to unit length to prevent vectors with large values from appearing to be more similar than they really are

#32D
pca32 = PCA(n_components=32)
compressedEmbeddings32 = pca32.fit_transform(fullEmbeddings)
compressedEmbeddings32 /= np.linalg.norm(compressedEmbeddings32, axis=1, keepdims=True) #normalized to unit length to prevent vectors with large values from appearing to be more similar than they really are
```

Normalization:
```
queryEmbedding /= np.linalg.norm(queryEmbedding, axis=1, keepdims=True)
```

Semantic Search Similarity
```
scores = embeddings @ queryEmbedding.T #scalar product for cosine similarity = how similar the query is to each document
                                       #.T for transposing matching dimensions for matrix multiplication
```

## 5. Analysis & Interpretation
semantic search vs keyword search

example queries and retrieved results

efficiency, scalability, limitations

## 6. How to run the project

First, install dependencies: 

```
python -m pip install --user sentence-transformers scikit-learn matplotlib numpy pandas
```
Then, run the demo:
```
python semanticCompression.py
```
## 7. Bibliography
[1]  James Jie Pan, Jianguo Wang & Guoliang Li, Survey of Vector Database Management Systems, 2023, https://arxiv.org/pdf/2310.14021, Last accessed: January 2026

[2] Cuzzocrea Alfredo, Vector Databases for Modelling, Managing and Querying Big Scientific Data: Models, Issues, Paradigms, 2025, https://dl.acm.org/doi/pdf/10.1145/3733723.3742469, Last accessed: January 2026

[3] Rahul Raja & Arpita Vats, Beyond Nearest Neighbors: Semantic Compression and Graph-Augmented Retrieval for Enhanced Vector Search, 2025, https://arxiv.org/pdf/2507.19715, Last accessed: January 2026

[4] OpenAI, ChatGPT, https://chatgpt.com/, Last accessed: January 2026

[5] Kaggle, AG News Classification Dataset, https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download, Last accessed: January 2026

[6] Hugging Face, sentence-transformers/all-MiniLM-L6-v2, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, Last accessed: January 2026
