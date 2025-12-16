"""
config.py - Central configuration for AI System.

This module stores all configurable constants related to:
- LLMs (chat, summarization)
- Embeddings
- Chunking and content limits
- Verification checks
- Dummy response simulator
- Qdrant Vector Database
"""

import os

# Model configuration::
LLM_CHAT_MODEL_NAME: str = "gemma3:1b"                  # Chatting model (1B faster version)
LLM_CHAT_TEMPERATURE: float = 0.75
LLM_SUMMARY_MODEL_NAME: str = "gemma3:1b"               # History Summarization model (1B faster version)
LLM_SUMMARY_TEMPERATURE: float = 0.5
EMB_MODEL_NAME: str = "mxbai-embed-large:latest"        # Embeddings model

# The max token count which shall be allowed after 'chat_history + input + context'.
MAX_CONTENT_SIZE: int = 14000


# Verification configuration:
#   - Whether to immediately verify the connection to
#   - the LLM models and the Embeddings models after initialization.
#   - Useful specifically for Ollama models, as they can be loaded on GPU or CPU.
VERIFY_LLM_CONNECTION: bool = False
VERIFY_EMB_CONNECTION: bool = False


# Document Chunking properties:
DOC_CHAR_LIMIT: int = 2000                              # Char limit for each doc.
DOC_OVERLAP_NO: int = 250                               # Char limit for chunk overlap.


# Document Retrieval properties:
DOC_TOKEN_SIZE: int = DOC_CHAR_LIMIT // 4               # Appx number of tokens in each doc.
DOCS_NUM_COUNT: int = 3000 // DOC_TOKEN_SIZE            # Max num of docs to retrieve.


# Qdrant Vector Database Configuration:
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME: str = "documents"

# Hybrid Retrieval Configuration (SPLADE Sparse + Dense DBSF Fusion + Voyage Reranking):
USE_HYBRID_RETRIEVAL: bool = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"
USE_SPLADE_SPARSE: bool = os.getenv("USE_SPLADE_SPARSE", "true").lower() == "true"
USE_MMR_DIVERSITY: bool = os.getenv("USE_MMR_DIVERSITY", "true").lower() == "true"
USE_VOYAGE_RERANKING: bool = os.getenv("USE_VOYAGE_RERANKING", "true").lower() == "true"
VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")

# Hybrid Retrieval Parameters (Default values):
HYBRID_FUSION_METHOD: str = "dbsf"              # "dbsf" (Density-Biased) or "rrf" (Reciprocal Rank)
HYBRID_MMR_DIVERSITY: float = 0.5               # Balance between relevance (0.0) and diversity (1.0)
HYBRID_DENSE_LIMIT: int = 20                    # Number of dense search results before fusion
HYBRID_CANDIDATES_LIMIT: int = 100              # Candidate limit for MMR pre-filtering
HYBRID_SPLADE_BATCH_SIZE: int = 6               # Batch size for SPLADE document processing
HYBRID_RETRIEVER_K: int = DOCS_NUM_COUNT        # Number of final documents to retrieve

# Dummy response mode properties:
TOKENS_PER_SEC: int = 50                                # num of tokens yielded per sec
BATCH_TOKEN_PS: int = 2                                 # num of tokens yielded in each batch
