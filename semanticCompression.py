import pandas as pd #for reading CSV files
import numpy as np #for numerical operations
import time
import matplotlib.pyplot as plt #for plotting graphs

from sentence_transformers import SentenceTransformer #for generating text embeddings
from sklearn.decomposition import PCA #for dimensionality reduction

#dataset loading
print("Loading dataset.csv file...", flush=True)
dataset = pd.read_csv("dataset.csv") 
documents = dataset['Description'].tolist()[:500] 
print(f"Loaded {len(documents)} documents succesfully from dataset.csv.\n", flush=True)


#embedding generation for full dimensions
print("Generating full embeddings...", flush=True) 
startTime = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2") #this is a pretrained open-source model with 384-dimensional embeddings from HuggingFace
fullEmbeddings = model.encode(documents, normalize_embeddings=True) #normalized for an easier cosine similarity calculation
timeForFull = time.time() - startTime
memoryForFull = fullEmbeddings.nbytes / (1024**2) #in MB

print(f"Full embeddings generated in {timeForFull:.2f} seconds.", flush=True)
print(f"Memory used for full embeddings: {memoryForFull:.2f} MB", flush=True)
print(f"Full embeddings shape is {fullEmbeddings.shape}", flush=True)
print("Expected shape: (500, 384)\nGot shape:", fullEmbeddings.shape, "\n", flush=True)


#semantic compression using PCA 
#PCA is am open-source linear dimensionality reduction technique that projects data onto a lower-dimensional subspace while still preserving as much variance as possible

#384-dimensional embeddings are reduced to 64 dimensions
print("Compressing embeddings from 384 dimensions to 64...", flush=True)
startTime = time.time()
pca64 = PCA(n_components=64)
compressedEmbeddings64 = pca64.fit_transform(fullEmbeddings)
compressedEmbeddings64 /= np.linalg.norm(compressedEmbeddings64, axis=1, keepdims=True) #normalized to unit length to prevent vectors with large values from appearing to be more similar than they really are
timeForPCA64 = time.time() - startTime
memoryForPCA64 = compressedEmbeddings64.nbytes / (1024**2) #in MB

print(f"Compressed embeddings (64 dimensions) generated in {timeForPCA64:.2f} seconds.", flush=True)
print(f"Memory used for compressed embeddings (64 dimensions): {memoryForPCA64:.2f} MB", flush=True)
print(f"Compressed embeddings shape: {compressedEmbeddings64.shape}", flush=True)
print("Expected shape: (500, 64)\nGot shape:", compressedEmbeddings64.shape, "\n", flush=True)

#384-dimensional embeddings are reduced to 32 dimensions
print("Compressing embeddings from 384 dimensions to 32...", flush=True)
startTime = time.time()
pca32 = PCA(n_components=32)
compressedEmbeddings32 = pca32.fit_transform(fullEmbeddings)
compressedEmbeddings32 /= np.linalg.norm(compressedEmbeddings32, axis=1, keepdims=True) #normalized to unit length to prevent vectors with large values from appearing to be more similar than they really are
timeForPCA32 = time.time() - startTime
memoryForPCA32 = compressedEmbeddings32.nbytes / (1024**2) #in MB

print(f"Compressed embeddings (32 dimensions) generated in {timeForPCA32:.2f} seconds.", flush=True)
print(f"Memory used for compressed embeddings (32 dimensions): {memoryForPCA32:.2f} MB", flush=True)
print(f"Compressed embeddings shape: {compressedEmbeddings32.shape}", flush=True)
print("Expected shape: (500, 32)\nGot shape:", compressedEmbeddings32.shape, "\n", flush=True)


#semantic search function
def search(query, embeddings, pca=None, top_k=5):
    queryEmbedding = model.encode([query], normalize_embeddings=True) #encoded and normalized for easier cosine similarity calculation
    
    if pca is not None: #if using compressed embeddings, the query embedding also needs to be compressed in the right pca form
        queryEmbedding = pca.transform(queryEmbedding)
        queryEmbedding /= np.linalg.norm(queryEmbedding, axis=1, keepdims=True)

    scores = embeddings @ queryEmbedding.T #scalar product for cosine similarity = how similar the query is to each document
                                           #.T for transposing matching dimensions for matrix multiplication
    topResults = np.argsort(scores.squeeze())[::-1][:top_k] #top k results in descending order
    return topResults

#recall function to see how many of the top results from full embeddings are also in the top results from compressed embeddings
def recall(fullResults, compressedResults):
    fullSet = set(fullResults)
    compressedSet = set(compressedResults)
    intersection = fullSet.intersection(compressedSet)
    recallValue = len(intersection) / len(fullSet)
    return recallValue

#main
query = "Schools near playgrounds with open green areas"

print("Top results using FULL embeddings:", flush=True)
for i in search(query, fullEmbeddings):
    print("-", documents[i][:150].replace("\n", " "), flush=True)

print("\nTop results using COMPRESSED embeddings(64 dimensions):", flush=True)
for i in search(query, compressedEmbeddings64, pca=pca64):
    print("-", documents[i][:150].replace("\n", " "), flush=True)

print("\nTop results using COMPRESSED embeddings(32 dimensions):", flush=True)
for i in search(query, compressedEmbeddings32, pca=pca32):
    print("-", documents[i][:150].replace("\n", " "), flush=True)

recall64 = recall(search(query, fullEmbeddings), search(query, compressedEmbeddings64, pca=pca64))
recall32 = recall(search(query, fullEmbeddings), search(query, compressedEmbeddings32, pca=pca32))
print("\nEfficiency analysis:", flush=True)
print(f"-Full 384 dimensions - Time: {timeForFull:.2f} seconds, Memory: {memoryForFull:.2f}MB", flush=True)
print(f"-Compressed 64 dimensions - Time: {timeForPCA64:.2f} seconds, Memory: {memoryForPCA64:.2f}MB, Recall: {recall64*100:.2f}%", flush=True)
print(f"-Compressed 32 dimensions - Time: {timeForPCA32:.2f} seconds, Memory: {memoryForPCA32:.2f}MB, Recall: {recall32*100:.2f}%\n", flush=True)

#plotting time, memory and recall
labels = ['Full 384 dimensions', 'Compressed 64 dimensions', 'Compressed 32 dimensions']
times = [timeForFull, timeForPCA64, timeForPCA32]
memories = [memoryForFull, memoryForPCA64, memoryForPCA32]
recalls = [1.0, recall64, recall32]  #full embeddings are 100% recall by definition

x = np.arange(len(labels)) #0 1 2
width = 0.3

figure, data = plt.subplots(figsize=(12,8))

data.bar(x - width/2, times, width, label='Time(seconds)', color='lightskyblue')
data.bar(x + width/2, memories, width, label='Memory(MB)', color='salmon')
data.plot(x, recalls, color='blueviolet', marker='o', label='Recall procentage', linewidth=2)

data.set_xticks(x) #0, 1, 2
data.set_xticklabels(labels) #Full, Compressed 64, Compressed 32

data.set_ylabel('Value')
data.set_title('Semantic Compression: Time, Memory & Recall Analysis')
data.legend(loc='upper right')

plt.show()
