# `RAG with Gemma-3`

This project is a **modular Retrieval-Augmented Generation (RAG) system** built with **Gemma 3** LLM (served locally via Ollama) and **Qdrant** vector database. It allows users to upload documents (PDF, TXT, Markdown, etc.) and chat with their content using natural language queries - all processed through a local setup for privacy and full control.

Designed with modularity and performance in mind, the system handles end-to-end workflows including file ingestion with hybrid embeddings (dense + sparse SPLADE vectors), vector storage in Qdrant, history summarization, document retrieval, context-aware response generation, and streaming replies to a frontend. It supports multi-file embeddings per user, persistent session history via Redis, document storage in PostgreSQL, and offers live document previews - making it a complete end-to-end RAG pipeline useful for educational and personal assistants.


# 📃 Index:
- [RAG with Gemma-3](#rag-with-gemma-3)
- [Project Details](#-project-details)
    - [Aim](#aim)
    - [Methodology](#methodology)
    - [Features](#features)
- [Tech Stack](#-tech-stack)
- [Installation](#%EF%B8%8F-installation)
    - [Virtual Environment](#virtual-environment)
    - [Docker](#-docker)
        1. [Using Docker Compose](#using-docker-compose)
        1. [Manual Docker Setup](#manual-docker-setup)
- [Configuration](#-configuration)
    - [Environment Variables](#environment-variables)
    - [Ollama Models](#ollama-models)
- [Troubleshooting](#-troubleshooting)
- [Future Work](#-future-work)
- [Contributions](#-contributions)
- [License](#-license)


# 🎯 Project Details:
## Aim
The core objective of this project is to build a **robust RAG system** with modern components and clean modular design and proper error handling.

## Methodology
1. Build a responsive UI in `Streamlit` allowing users to upload documents, get previews to ensure correctness, and interact with them.
2. Use `FastAPI` to build a backend that handles file uploads, document processing, and streaming LLM responses.
3. Implement modular `LLM System` using `LangChain` components for chains, embeddings, retrievers, vector storage, history management, and overall LLM orchestration.
4. Integrate locally hosted `Gemma-3` LLM via `Ollama` for local inference with no API keys required.
5. Implement **hybrid embeddings**: dense vectors using `mxbai-embed-large` and sparse vectors using **SPLADE** for improved retrieval accuracy.
6. Use `Qdrant` for efficient vector storage, hybrid search, and user-specific document storage and retrieval.
7. Use `PostgreSQL` for user management, authentication, and data control.
8. Use `Redis` for persistent session history and chat context management.
9. Create a unified `Docker` setup supporting both development and production environments with `docker-compose` for orchestration.
10. Enforce context-aware responses with system prompts that prioritize document content over LLM training knowledge.


## Features

- **Context-Aware RAG**: 
    + System enforces responses based exclusively on document context, preventing hallucinations and ensuring factual accuracy.
    + Uses hybrid retrieval combining dense DBSF search with sparse SPLADE vectors for superior document matching.

- **User Authentication & Data Management**:
    + Authenticate users via `PostgreSQL` database with secure password management.
    + Track user uploaded files and corresponding metadata in database.
    + Persistent session history stored in `Redis` for context-aware interactions.
    + Allow users to delete documents and manage session history.

- **Advanced Document Ingestion**:
    + Support multi-file uploads per user with automatic processing.
    + Generate **hybrid embeddings**: dense vectors (1024-dim) and sparse vectors (SPLADE).
    + Store documents in `Qdrant` with full metadata (source, page number, timestamp).
    + Display live document previews with 10-minute caching.

- **Intelligent Retrieval**:
    + Hybrid retriever combining dense similarity search and sparse BM25-style search.
    + Density-Biased Fusion (DBSF) score fusion for optimal ranking.
    + Maximal Marginal Relevance (MMR) for result diversity.
    + Efficient vector storage and retrieval via Qdrant.

- **Streaming LLM Responses**:
    + Built-in **FastAPI** backend with Server-Sent Events (SSE) for real-time streaming.
    + Stream LLM responses chunk-by-chunk for responsive UX.
    + Display retrieved documents and metadata for verification of responses.

- **LLM System Architecture**:
    + Modular design using `LangChain` for:
        - **Document Ingestion**: Load and chunk documents efficiently.
        - **Vector Embedding**: Dense embeddings via `mxbai-embed-large` and sparse via SPLADE.
        - **History Management**: Summarize and manage session history in Redis.
        - **Document Retrieval**: Hybrid search combining dense and sparse vectors.
        - **Response Generation**: Context-aware responses from Gemma-3 via Ollama.
        - **Tracing**: Optional LangSmith integration for debugging and monitoring.

- **Unified Containerization**:
    + Single `Dockerfile` supporting both development and production environments.
    + `docker-compose.yml` orchestrates all services: Qdrant, Ollama, PostgreSQL, Redis, FastAPI, Streamlit.
    + Easy environment switching with environment variables (no separate deploy files).
    + Production-ready with all services networked and persistent volumes.


# 🧑‍💻 Tech Stack
- 🦜 **LangChain** - LLM orchestration and chain management
- ⚡ **FastAPI** - Backend API with SSE streaming
- 👑 **Streamlit** - Frontend UI
- 🐋 **Docker & Docker Compose** - Containerization and orchestration
- 🦙 **Ollama** - Local LLM serving
    - **Gemma 3** (1B) - Primary LLM for responses
    - **mxbai-embed-large** - Dense embeddings (1024-dim)
    - **SPLADE** (via FastEmbed) - Sparse embeddings
- ♾️ **Qdrant** - Vector database with hybrid search support
- 🐘 **PostgreSQL** - User and document metadata storage
- 🔴 **Redis** - Session history and chat context caching
- 🛠️ **LangSmith** - Optional LLM tracing and debugging
- 🔐 **bcrypt** - Password hashing and security


# 🛠️ Installation
There are two ways to run this project: using [**Docker Compose**](#using-docker-compose) (recommended) or using a [**Virtual Environment**](#virtual-environment).

## Docker Compose (Recommended)

This is the easiest way to run the entire system with all required services.

### Prerequisites:
- Docker and Docker Compose installed
- Ollama running with models pre-pulled:
  ```bash
  ollama pull gemma3:1b
  ollama pull mxbai-embed-large:latest
  ```

### Steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Bbs1412/rag-with-gemma3.git
    cd rag-with-gemma3
    ```

2. Start all services with docker-compose:
    ```bash
    docker-compose up -d
    ```

3. Access the applications:
    - **Streamlit Frontend**: http://localhost:8501
    - **FastAPI Backend**: http://localhost:8000
    - **FastAPI Docs**: http://localhost:8000/docs
    - **Qdrant Console**: http://localhost:6333/dashboard

4. Stop services:
    ```bash
    docker-compose down
    ```

### Manual Docker Setup

If you prefer to build and run containers manually:

1. **Build the Docker image**:
    ```bash
    docker build -t chat-with-data:latest .
    ```

2. **Start infrastructure services** (Qdrant, Ollama, PostgreSQL, Redis):
    ```bash
    docker-compose up -d qdrant ollama postgres redis
    ```

3. **Run the application container**:
    ```bash
    docker run -d --name chat-app \
        --network chat-network \
        -e QDRANT_HOST=qdrant \
        -e REDIS_HOST=chat-redis \
        -e POSTGRES_HOST=chat-postgres \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=chat_db \
        -p 8000:8000 \
        -p 8501:8501 \
        chat-with-data:latest
    ```

## Virtual Environment

For local development without Docker:

1. Clone the repository:
    ```bash
    git clone https://github.com/Bbs1412/rag-with-gemma3.git
    cd rag-with-gemma3
    ```

2. Create and activate a virtual environment:
    ```bash
    # Create environment
    python -m venv venv
    
    # Activate environment
    source venv/bin/activate  # On Linux/macOS
    # or
    venv\Scripts\activate  # On Windows

    # Install dependencies
    pip install -r requirements.txt
    ```

3. Ensure Ollama is running with required models:
    ```bash
    ollama serve  # In a separate terminal
    ```

4. (Optional) Configure LangSmith tracing by creating `.env` in the `server` directory:
    ```ini
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    LANGCHAIN_API_KEY=<your_api_key>
    LANGCHAIN_PROJECT=rag-with-gemma3
    ```

5. Start the FastAPI server:
    ```bash
    cd server
    uvicorn server:app --reload --port 8000
    ```

6. In another terminal, start the Streamlit app:
    ```bash
    cd ..
    streamlit run app.py
    ```

1. Access the applications:
    - **Streamlit Frontend**: http://localhost:8501
    - **FastAPI Backend**: http://localhost:8000
    - **FastAPI Docs**: http://localhost:8000/docs


# ⚙️ Configuration

## Environment Variables

The system reads the following environment variables for configuration:

### Qdrant Configuration
- `QDRANT_HOST` (default: `localhost`) - Qdrant server hostname
- `QDRANT_PORT` (default: `6333`) - Qdrant server port

### Redis Configuration
- `REDIS_HOST` (default: `localhost`) - Redis server hostname
- `REDIS_PORT` (default: `6379`) - Redis server port

### PostgreSQL Configuration
- `POSTGRES_HOST` (default: `localhost`) - PostgreSQL server hostname
- `POSTGRES_PORT` (default: `5432`) - PostgreSQL server port
- `POSTGRES_USER` (default: `postgres`) - PostgreSQL username
- `POSTGRES_PASSWORD` (default: `postgres`) - PostgreSQL password
- `POSTGRES_DB` (default: `chat_db`) - PostgreSQL database name

### LangSmith Tracing (Optional)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing
- `LANGCHAIN_ENDPOINT` - LangSmith API endpoint
- `LANGCHAIN_API_KEY` - LangSmith API key
- `LANGCHAIN_PROJECT` - Project name for tracing

## Ollama Models

To change LLM or Embedding models, edit [`./server/llm_system/config.py`](./server/llm_system/config.py):

- **LLM Model**: `LLM_MODEL` - Currently `gemma3:1b`
- **Embedding Model**: `EMBEDDING_MODEL` - Currently `mxbai-embed-large:latest`
- **Max Context**: `MAX_CONTEXT_LENGTH` - Currently `2048` tokens
- **Temperature**: `LLM_TEMPERATURE` - Currently `0.7`

### Inference Device Configuration

In [`./server/llm_system/core/database.py`](./server/llm_system/core/database.py):
- Change `num_gpu` parameter to control GPU offloading:
  - `0` = 100% CPU
  - `-1` = 100% GPU
  - `n` = Offload `n` layers to GPU
- Ollama automatically manages GPU if the parameter is omitted

# 🔧 Troubleshooting

## Redis Connection Issues
- **Error**: `'NoneType' object has no attribute 'lrange'`
- **Solution**: Ensure `REDIS_HOST` environment variable is set correctly (e.g., `chat-redis` in Docker)

## Ollama Connection Issues
- **Error**: Connection refused to `http://ollama:11434`
- **Solution**: Ensure Ollama is running and accessible. In docker-compose, check that the `ollama` service is running: `docker-compose ps`

## Qdrant Connection Issues
- **Error**: Connection refused to Qdrant
- **Solution**: Verify `QDRANT_HOST` and `QDRANT_PORT` environment variables. In docker-compose, the hostname is `qdrant` not `localhost`

## Streamlit Not Loading
- **Solution**: Clear Streamlit cache with: `streamlit cache clear`

## Port Already in Use
- **Solution**: Either stop the conflicting service or change ports in `docker-compose.yml`

