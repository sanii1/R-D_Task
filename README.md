# Agentic RAG Application (HyDE-Enhanced)

## Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** that can ingest documents, answer user questions using a local Large Language Model (LLM), and evaluate its own responses. The system is enhanced with **HyDE (Hypothetical Document Embeddings)** to improve retrieval quality for complex or low-overlap queries.

The entire application is **containerized using Docker**, making it easy to deploy and run with minimal setup.

---

## Key Features

* **Document Ingestion**: Supports PDF and text files with automatic chunking and embedding
* **RAG Pipeline**: Retrieval + Generation workflow built using LangGraph
* **HyDE Retrieval**: Improves recall and precision by expanding query semantics
* **Local LLM Inference**: Uses Ollama for running models locally
* **Vector Database**: Weaviate for efficient similarity search
* **Evaluation Framework**: Metric-based evaluation (relevance, faithfulness, hallucination risk)
* **Dockerized Setup**: Consistent runtime across environments

---

## Tech Stack

* **Language**: Python 3.11
* **API Framework**: FastAPI
* **Orchestration**: LangChain, LangGraph
* **Vector Store**: Weaviate
* **LLM Runtime**: Ollama (local models like LLaMA)
* **Evaluation**: Ragas-style evaluation logic
* **Containerization**: Docker & Docker Compose

---

## Architecture Overview

High-level flow:

User → FastAPI → RAG Pipeline → Weaviate → Ollama → Response

HyDE (optional) is used during retrieval to generate hypothetical answers that improve vector search results.

---

## Project Structure

```
project-root/
│
├── app/
│   ├── core/
│   │   ├── config.py        # Environment & settings
│   │   ├── db.py            # Weaviate client
│   │   ├── llm.py           # LLM & embedding utilities
│   │   ├── ingestion.py     # Document ingestion logic
│   │   └── rag.py           # RAG + HyDE pipeline
│   │
│   ├── schemas.py           # API schemas
│   └── main.py              # FastAPI entry point
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── eval.py                  # Evaluation pipeline
└── README.md
```

---

## Setup & Installation

### Prerequisites

* Docker & Docker Compose
* Ollama installed locally
* A local model pulled in Ollama (example):

```bash
ollama pull llama3
```

---

## Running the Application

### 1. Start Services

From the project root:

```bash
docker compose up --build
```

This starts:

* Weaviate on `http://localhost:8080`
* FastAPI app on `http://localhost:8000`

---

### 2. Ingest Documents

Use the `/ingest` endpoint to load documents into Weaviate.

Example:

```bash
POST /ingest?file_path=docs/sample.pdf
```

Documents are automatically chunked and embedded.

---

### 3. Ask Questions

Use the `/chat` endpoint:

```json
{
  "question": "What is this document about?",
  "use_hyde": true
}
```

The system retrieves relevant context and generates an answer using the local LLM.

---

## Evaluation

The project includes an evaluation pipeline that measures:

* Relevance
* Factual Accuracy / Faithfulness
* Hallucination Risk
* Estimated Cost

Two modes are compared:

1. **Standard RAG**
2. **HyDE-Enhanced RAG**

HyDE shows improved recall and precision, especially for complex queries with low lexical overlap.

---

## Limitations & Notes

* Full-scale runtime evaluation was limited due to local hardware constraints when running multiple containerized services simultaneously.
* Evaluation metrics are indicative and demonstrate methodology rather than absolute benchmarks.
* Security, authentication, and multi-collection routing are not implemented in this version.

---

## Future Improvements

* Add reranking (cross-encoder)
* Streaming responses
* Auth & user management
* Multi-document and multi-collection routing
* GPU-backed inference

---

## Conclusion

This project demonstrates a complete, modular, and extensible **Agentic RAG system** with modern retrieval enhancements and evaluation practices. It focuses on correctness, architecture, and real-world applicability rather than simple prompt-based Q&A.

---

## Author

Muhammed Saneen

---

*This repository is intended for technical evaluation, learning, and demonstration purposes.*
