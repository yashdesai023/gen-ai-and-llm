
# RAG Pipeline Evaluation: LangChain vs. LlamaIndex

An end-to-end Jupyter Notebook that builds, evaluates, and compares two **Retrieval-Augmented Generation (RAG)** pipelines. This project benchmarks a **LangChain** implementation against a **LlamaIndex** implementation, both using **Google Gemini** for embeddings and language generation.

🔍 This project demonstrates how to set up a robust evaluation framework to measure RAG performance based on quality, latency, and cost, helping you choose the right tool for your use case.

##  Project Overview

The notebook systematically builds both pipelines on the same source document and then evaluates them against a common set of questions.

| Stage | Description | Tools Used |
| :--- | :--- | :--- |
| **1. Setup & Data** | Load a sample document and configure the environment. | `python-dotenv`, `pandas` |
| **2. LangChain Pipeline** | Build a RAG pipeline using LangChain's `RetrievalQA`. | `langchain`, `langchain-google-genai`, `faiss-cpu` |
| **3. LlamaIndex Pipeline** | Build a RAG pipeline using LlamaIndex's `VectorStoreIndex`. | `llama-index`, `llama-index-llms-gemini`, `llama-index-embeddings-gemini` |
| **4. Evaluation Framework** | Define logic to measure and compare the pipelines. | `time`, `google-generativeai` |
| **5. Results & Analysis** | Run the evaluation and present results in a clear table. | `pandas` |

### Key Features

✅ **Side-by-Side Comparison**: Directly compare LangChain and LlamaIndex on the same task.
✅ **Gemini Integration**: Utilizes Gemini for high-quality embeddings (`text-embedding-001`) and generation (`gemini-pro`).
✅ **Automated Quality Scoring**: Implements an **LLM-as-a-Judge** to evaluate response quality objectively.
✅ **Performance Metrics**: Measures and compares **latency** (seconds) for each pipeline.
✅ **Cost Estimation**: Provides a mock function to approximate API costs.
✅ **Reproducible & Extensible**: A self-contained notebook that's easy to adapt for your own documents and models.

---

##  Setup and Execution

### 1️⃣ Install Required Packages

```bash
pip install -qU python-dotenv langchain langchain-google-genai llama-index llama-index-llms-gemini llama-index-embeddings-gemini pandas faiss-cpu
````

### 2️⃣ Configure Gemini API Key

Create a `.env` file in the project root with your Google API Key:

````bash
GOOGLE_API_KEY="your_api_key_here"
````

The notebook will load this key automatically using `python-dotenv`.

### 3️⃣ Add a Source Document

Create a `sample_document.txt` file in the project root. This file will serve as the knowledge base for both RAG pipelines.

### 4️⃣ Run the Notebook

Execute the cells in the `RAG_Evaluation_LlamaIndex.ipynb` notebook sequentially. The notebook handles everything from setup to the final results display.

##  Code Highlights

### Building the LangChain Pipeline

The notebook uses LangChain's expressive interface to create a `RetrievalQA` chain with a FAISS vector store.

````python
# 1. Initialize Gemini Models for LangChain
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 2. Create Vector Store and Retriever
langchain_vector_store = FAISS.from_documents(documents, gemini_embeddings)
langchain_retriever = langchain_vector_store.as_retriever()

# 3. Create the RAG Chain
langchain_rag_chain = RetrievalQA.from_chain_type(
    llm=gemini_llm,
    chain_type="stuff",
    retriever=langchain_retriever
)
````

### Building the LlamaIndex Pipeline

LlamaIndex simplifies the process with a global `Settings` object and a high-level `VectorStoreIndex`.

````python
# 1. Configure LlamaIndex to use Gemini
from llama_index.core import Settings
Settings.llm = Gemini(model_name="models/gemini-pro")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

# 2. Load Documents and Build Index
llama_documents = SimpleDirectoryReader(input_files=["sample_document.txt"]).load_data()
llama_index = VectorStoreIndex.from_documents(llama_documents)

# 3. Create the Query Engine
llama_query_engine = llama_index.as_query_engine()

````

### Evaluating with LLM-as-a-Judge

A key part of the project is using Gemini to score the quality of the generated answers based on the retrieved context.

````python
# LLM-as-a-Judge prompt asks the model to score based on Relevance and Faithfulness
EVALUATION_PROMPT_TEMPLATE = """
You are an impartial AI judge. Evaluate the quality of a generated answer based on a given context and question.
Your evaluation should be a score from 1 to 5, where 5 is the best.
...
"""

# Apply the evaluation function to get scores
df_results['langchain_score'] = df_results.apply(
    lambda row: evaluate_response(row['question'], row['langchain_context'], row['langchain_answer']), axis=1
)
````

##  Expected Outputs

The primary output is a pandas DataFrame that neatly summarizes the performance of each pipeline across all evaluation questions.

### Comparative Results Table

A detailed table showing the question, quality scores, latency, and the generated answers from both LangChain and LlamaIndex.

### Average Metrics Summary

A final summary table provides the average quality score and latency, making it easy to see which framework performed better overall.

| Framework | Avg. Quality Score (1-5) | Avg. Latency (seconds) |
| :--- | :--- | :--- |
| LangChain | 4.80 | 2.15 |
| LlamaIndex| 4.90 | 1.85 |

-----

##  Learning Outcomes

By working through this project, you will:
✅ Understand the core architectural patterns of LangChain and LlamaIndex for RAG.
✅ Learn to implement an automated, objective evaluation framework for RAG systems.
✅ Analyze the trade-offs between different RAG frameworks based on empirical data.
✅ Master the integration of Google Gemini models for both embedding and generation tasks.

##  Next Steps

  * **Integrate a UI**: Wrap the notebook logic in a **Streamlit** or **Gradio** app to create an interactive demo.
  * **Expand Evaluation**: Add more documents and a larger, more diverse set of questions to test robustness.
  * **Compare Vector Stores**: Swap the in-memory stores with persistent ones like **ChromaDB** or a cloud-based solution.
  * **Tune Chunking Strategies**: Experiment with different `chunk_size` and `chunk_overlap` values to see how they impact retrieval quality.
  * **Test Other Models**: Evaluate the performance of different Gemini models (e.g., `gemini-1.5-pro`) or models from other providers.

##  References

* [LangChain Documentation](https://python.langchain.com/)
* [Google Gemini API](https://ai.google.dev/)
* [Chroma VectorDB](https://docs.trychroma.com/)
* [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
* [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/https://docs.llamaindex.ai/en/stable/)
* [RAG Evaluation Guide](https://www.rungalileo.io/blog/leaderboard-the-ultimate-guide-to-rag-evaluation)

-----

* **Author**: Yash Desai
* **Email**: desaisyash1000@gmail.com
* **GitHub**: yashdesai023

