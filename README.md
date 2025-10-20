# Generative AI & LLM Systems

This repository showcases my **Generative AI & LLM Systems** journey — focused on building, evaluating, and deploying Large Language Model (LLM) applications.  
Each project emphasizes **hands-on experimentation**, **framework comparison**, and **real-world deployable outcomes** using **Gemini APIs**, **LangChain**, and **LlamaIndex**.

---

## 📘 Overview

| Phase | Focus Area | Key Outputs | Repository Path |
|-------|-------------|--------------|------------------|
| 1. Transformer Architecture & Tokenization | Text to Embedding Conversion | Embedding Visualizer | `/transformers_tokenization` |
| 2. LangChain & Vector Databases | Retrieval-Augmented Generation (RAG) System | DocChat RAG Bot | `/rag_chatbot_langchain` |
| 3. LlamaIndex & RAG Evaluation | RAG Comparison & Performance Analysis | RAG Evaluation Notebook | `/rag_evaluation_llamaindex` |
| 4. Function Calling & AI Agents | Tool-Using AI Assistant | Function-Calling Agent | `/function_calling_agent` |
| 5. Multi-Agent Automation | Collaborative AI Agents | Research Crew App | `/multiagent_crew_app` |
| 6. App Deployment | Streamlit + Gemini API Deployments | RAG Chatbot App, AI Agent App | `/deployment_apps` |

---

## 🧩 Tools & Frameworks Used

- **LLM Frameworks:** LangChain, LlamaIndex, CrewAI, LangGraph  
- **Models & APIs:** Google Gemini (Text & Embedding APIs)  
- **Vector Databases:** Chroma, FAISS  
- **Frontend Deployment:** Streamlit, Gradio, Hugging Face Spaces  
- **Evaluation & Analytics:** FAISS similarity, latency comparison  
- **Languages:** Python 3.10  
- **Environment Management:** dotenv, GitHub, VS Code  

---

## 🧠 Highlight Projects

### 1. Transformer & Tokenization — *Embedding Visualizer*
**Goal:** Understand how text is converted to vector embeddings using Transformers.

**Tech Stack:** Hugging Face Transformers, Gemini Embedding API, Matplotlib, PCA  
**Key Steps:**
- Tokenization using `AutoTokenizer`  
- Embedding generation using Gemini model `text-embedding-004`  
- PCA reduction and 2D visualization of semantic relationships  

**Files:** `Transformer_Tokenization.ipynb`  
**Visuals:** PCA Plot of Text Embeddings, Tokenization Breakdown  
**Outcome:** Learned how raw text translates into numerical embeddings forming the foundation for RAG systems.  

---

### 2. LangChain RAG System — *DocChat RAG Bot*
**Goal:** Build a document-based Q&A system using LangChain and Gemini embeddings.  

**Tech Stack:** LangChain, ChromaDB, Gemini API, Streamlit  
**Pipeline:**
1. Upload PDF/Text  
2. Chunk + Embed using `GoogleGenerativeAIEmbeddings`  
3. Store vectors in Chroma  
4. Retrieve context and generate response using Gemini model  

**Files:** `RAG_Chatbot_LangChain.ipynb`  
**Visuals:** RAG Pipeline Diagram, Chat Interface Screenshot  
**Outcome:** Fully working RAG pipeline — capable of answering document-based queries locally.  

---

### 3. RAG Evaluation — *LangChain vs LlamaIndex*
**Goal:** Compare retrieval performance, relevance, and latency between two popular RAG frameworks.  

**Tech Stack:** LlamaIndex, LangChain, Gemini API, FAISS  
**Key Experiments:**
- Same document processed through both pipelines  
- Retrieval quality comparison via similarity scores  
- Response evaluation by relevance and generation latency  

**Files:** `RAG_Evaluation_LlamaIndex.ipynb`  
**Visuals:** Framework Comparison Chart, Response Examples  
**Outcome:** Insightful analysis of both RAG frameworks — helping decide which suits different production scenarios.  

---

### 4. Function Calling & AI Agents — *AI Assistant Agent*
**Goal:** Build an AI system that can autonomously call tools and APIs.  

**Tech Stack:** LangChain Tools, Gemini Function Calling, DuckDuckGoSearchResults  
**Pipeline:**
- Primary agent (LLM) handles queries  
- Function-calling agent executes API tasks (search, calculator, weather, etc.)  
- Collaborative response synthesis  

**Files:** `FunctionCalling_AgentApp.ipynb`  
**Visuals:** Tool Use Flow, Query → API → Response Mapping  
**Outcome:** Created a modular AI assistant that blends reasoning + tool execution for dynamic responses.  

---

### 5. Multi-Agent System — *Research Crew App*
**Goal:** Demonstrate a multi-agent collaboration workflow using CrewAI or LangGraph.  

**Tech Stack:** CrewAI, LangGraph, Gemini API  
**Pipeline Roles:**
1. **Researcher:** Searches the web for content  
2. **Summarizer:** Extracts and compresses key points  
3. **Reporter:** Generates a formatted summary  

**Files:** `MultiAgent_CrewApp.ipynb`  
**Visuals:** Agent Workflow Diagram, Role Interaction Flow  
**Outcome:** Built an automated information processing chain — showing coordinated agent reasoning and task delegation.  

---

### 6. Deployment — *RAG Chatbot App & AI Agent App*
**Goal:** Deploy both LLM apps using Streamlit + Gemini APIs on free cloud platforms.  

**Tech Stack:** Streamlit, LangChain, Gemini API, Hugging Face Spaces / Streamlit Cloud  
**Apps:**
- **RAG_Chatbot_App:** Document-based Q&A chatbot powered by Gemini embeddings.  
- **AI_Agent_App:** Function-calling assistant with web search capability.  

**Deployment Flow:**
1. Build Streamlit UI  
2. Configure Gemini API  
3. Host via Hugging Face or Streamlit Cloud  

**Files:**  
- `/RAG_Chatbot_App/app.py`  
- `/AI_Agent_App/app.py`  
**Visuals:** Web App Interfaces, Deployment Screenshot  
**Outcome:** Fully hosted, public-facing LLM applications demonstrating real deployment capability.

---

## 📈 Insights Summary

- Gained complete exposure to **LLM lifecycle** — from tokenization → embedding → retrieval → reasoning → deployment.  
- Mastered **LangChain, LlamaIndex, and CrewAI** through project-based implementation.  
- Built **deployable AI systems** capable of retrieval, reasoning, and multi-agent automation.  
- Created portfolio-ready, production-style case studies demonstrating **applied Generative AI engineering**.  

---

## 📦 Setup Instructions

```bash
git clone https://github.com/yashdesai023/genai-llm-systems.git
cd genai-llm-systems
pip install -r requirements.txt
````

Run notebooks via:

````bash
jupyter notebook
````

Or launch apps locally:

````bash
cd RAG_Chatbot_App
streamlit run app.py
````

---

##  **Author:**
### Hi, I'm Yash Desai  

**Generative AI & LLM Engineer | Java • Python • Spring Boot | Exploring DevOps & SDET | Building Scalable AI Solutions**  

With a strong foundation in **Java backend engineering** and a growing expertise in **Generative AI**, I specialize in building intelligent, production-ready applications that merge the **robustness of enterprise systems** with the **innovation of Large Language Models (LLMs)**.  

I thrive on designing solutions that are **scalable, impactful, and deployment-ready**—bridging backend reliability with AI innovation to deliver **real-world value** for businesses and communities. 

#### 🔥 **About Me**
* 🎓 CSE Student specializing in Artificial Intelligence
* 💻 Experienced in Java | Spring Boot | SQL | Backend Systems
* 🤖 Building applied Generative AI & LLM projects with Python, LangChain & TensorFlow
* ⚙️ Exploring DevOps & SDET practices (CI/CD, Docker, Jenkins, Testing)
* 🌍 Goal: To stand at the intersection of enterprise engineering & AI by delivering scalable, intelligent solutions

<p align="center">
  <a href="https://www.linkedin.com/in/yash-s-desai-/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/yashdesai023">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="mailto:desaisyash1000@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/>
  </a>
</p>

