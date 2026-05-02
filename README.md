```markdown
# 🧠 DocuMind — AI PDF Assistant

> Ask anything from any document. Powered by RAG + LangGraph + Groq.

## 🚀 Tech Stack
- **Frontend** — React.js
- **Backend** — FastAPI
- **Vector DB** — ChromaDB
- **Embeddings** — HuggingFace (all-MiniLM-L6-v2)
- **LLM** — LLaMA 3.3 70B via Groq API
- **Orchestration** — LangGraph

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/T-Sid-1025/DocuMind.git
cd DocuMind
```

### 2. Create `.env` file in root
```
GROQ_API_KEY=your_key_here
```

### 3. Backend Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn langchain langchain-community chromadb sentence-transformers groq langgraph pypdf python-dotenv
uvicorn backend.app:app --reload
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm start
```

## ✨ Features
- 📄 PDF Upload & Processing
- 🔍 Semantic Search with ChromaDB
- 🧠 Multi-turn Conversation Memory
- 📍 Source Highlighting with Page Numbers
- 👁️ In-app PDF Preview
- 💾 Chat History saved in localStorage
- 🎤 Voice Input Support
- 🔀 LangGraph Pipeline with HITL Escalation

## 🏗️ Project Structure
```
DocuMind/
├── backend/
│   ├── app.py          # FastAPI server
│   ├── ingestion.py    # PDF processing & chunking
│   ├── rag_pipeline.py # Retrieval + Generation
│   ├── graph.py        # LangGraph orchestration
│   └── config.py       # Embedding model config
├── frontend/
│   └── src/
│       ├── App.js      # Main React component
│       └── App.css     # Styling
├── data/               # Uploaded PDFs (gitignored)
├── chroma_db/          # Vector DB (gitignored)
└── .env                # API keys (gitignored)
```

## 👨‍💻 Author
**Siddhant Tagare** — Agentic AI Intern @ Innomatics Research Labs, Pune

[![GitHub](https://img.shields.io/badge/GitHub-T--Sid--1025-black?logo=github)](https://github.com/T-Sid-1025)
```
