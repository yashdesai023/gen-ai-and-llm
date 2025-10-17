

#  DocChat RAG Bot — LangChain + Chroma + Gemini

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that allows users to **upload a PDF or text document**, store its embeddings in a **vector database (Chroma)**, and **chat with the content** using **Google Gemini models**.

> 🔍 This project demonstrates how to combine **LangChain**, **Chroma**, and **Gemini embeddings** to build a local knowledge chatbot.

---

##  Project Overview

| Stage | Description | Tools |
|--------|--------------|-------|
| **1. Document Loading** | Load PDF or text documents | `PyPDFLoader`, `TextLoader` |
| **2. Text Splitting** | Break text into manageable chunks | `RecursiveCharacterTextSplitter` |
| **3. Embedding Generation** | Convert text chunks → numerical vectors | `GoogleGenerativeAIEmbeddings (text-embedding-004)` |
| **4. Vector Database** | Store & retrieve embeddings efficiently | `Chroma` |
| **5. RAG Chain** | Retrieve relevant chunks + generate answers | `RetrievalQA`, `ChatGoogleGenerativeAI` |

---

##  Key Features

✅ Upload and process your own documents  
✅ Generate embeddings using **Gemini text-embedding-004**  
✅ Store and retrieve chunks with **ChromaDB**  
✅ Ask context-aware questions about your file  
✅ End-to-end pipeline using **LangChain’s RetrievalQA**  


---

##  Setup Instructions

### 1️⃣ Install Required Packages

```bash
pip install langchain langchain-google-genai chromadb faiss-cpu PyPDF2 python-dotenv
```

---

### 2️⃣ Configure Gemini API Key

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

Then load it inside the notebook:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

### 3️⃣ Load the Document

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample_doc.pdf")
documents = loader.load()

print("Loaded documents:", len(documents))
```

---

### 4️⃣ Split Text into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.split_documents(documents)
print("Number of chunks:", len(docs))
```

---

### 5️⃣ Create Embeddings using Gemini

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

---

### 6️⃣ Store Embeddings in ChromaDB

```python
from langchain.vectorstores import Chroma

vectordb = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db"
)
vectordb.persist()
print("Vector database created successfully!")
```

---

### 7️⃣ Build the RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    retriever=retriever,
    return_source_documents=True
)
```

---

### 8️⃣ Ask Questions about the Document

```python
query = "What is the main topic discussed in the document?"
result = qa_chain.invoke({"query": query})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata)
```

---

### 9️⃣ (Optional) Visualize Embedding Clusters

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

vectors = [embeddings.embed_query(doc.page_content) for doc in docs[:200]]
reduced = PCA(n_components=2).fit_transform(vectors)

plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("Document Embeddings Visualization (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

---

##  Outputs

| Output                    | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
|  **Contextual Answers** | Gemini generates answers based on retrieved document chunks |
|  **ChromaDB Storage**  | Embeddings stored for retrieval                             |
|  **PCA Plot**           | 2D visualization of document embeddings                     |
|  **Console Logs**       | Chunk count, retrieval confirmation, and Gemini response    |

---

##  Visual Placeholders

| Visual                   | Description                      |
| ------------------------ | -------------------------------- |
|  **doc_upload.png**   | Uploading document interface (Not Uploaded Yet)    |
|  **vector_db.png**     | Visualization of vector database (Not Uploaded Yet) |
|  **chat_output.png**   | Example question-answer exchange (Not Uploaded Yet) |
|  **embedding_viz.png** | PCA visualization of embeddings (Not Uploaded Yet) |

---

## 🧠 Learning Outcomes

✅ Understand RAG (Retrieval-Augmented Generation) architecture
✅ Learn how embeddings power document-based QA
✅ Use Gemini for both embeddings & generative responses
✅ Implement Chroma as a lightweight, local vector database
✅ End-to-end pipeline from document → embeddings → chatbot

---

## 🚀 Next Steps

* Integrate with **Streamlit / Gradio** UI
* Add support for multiple documents
* Use **FAISS** for scalable retrieval
* Implement **Hybrid Search (BM25 + Embeddings)**
* Add metadata filtering for domain-specific queries

---

## 📎 References

* [LangChain Documentation](https://python.langchain.com/)
* [Google Gemini API](https://ai.google.dev/)
* [Chroma VectorDB](https://docs.trychroma.com/)
* [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---

* **Author:** Yash Desai
* **Email:** desaisyash1000@gmail.com 
* **GitHub:** yashdesai023

