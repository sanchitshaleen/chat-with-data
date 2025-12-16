# This file is for deployment using Ollama with Gemma2.5 and Qdrant

""" Database Module for LLM System
- Contains the `VectorDB` class to manage a vector database using Qdrant and Ollama embeddings.
- Provides methods to initialize the database, retrieve embeddings, and perform similarity searches.
"""

import time
from typing import Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import ConfigurableField

# For type hinting
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from logger import get_logger
log = get_logger(name="core_database")


class VectorDB:
    """A class to manage the vector database using Qdrant and Google Generative AI embeddings.

    Args:
        embed_model (str): The name of the Google Generative AI embeddings model to use.
        retriever_num_docs (int): Number of documents to retrieve for similarity search.
        verify_connection (bool): Whether to verify the connection to the embeddings model.
        qdrant_host (str): Qdrant server host. Defaults to localhost.
        qdrant_port (int): Qdrant server port. Defaults to 6333.
        collection_name (str): Name of the Qdrant collection. Defaults to "documents".

    ## Functions:
        + `get_embeddings()`: Returns the Google Generative AI embeddings model.
        + `get_vector_store()`: Returns the Qdrant vector store.
        + `get_retriever()`: Returns the retriever configured for similarity search.
    """

    def __init__(
        self, embed_model: str,
        retriever_num_docs: int = 5,
        verify_connection: bool = False,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents"
    ):
        self.retriever_num_docs = retriever_num_docs
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

        log.info(
            f"Initializing VectorDB with Qdrant at '{self.qdrant_url}', "
            f"embeddings='{embed_model}', collection='{collection_name}', k={retriever_num_docs} docs."
        )

        # Load Ollama embeddings (mxbai-embeddings-large model)
        self.embeddings = OllamaEmbeddings(base_url="http://chat-ollama:11434", model=embed_model)

        if verify_connection:
            try:
                self.embeddings.embed_documents(['a'])
                log.info(f"Embeddings model '{embed_model}' initialized and verified.")

            except Exception as e:
                log.error(f"Failed to initialize Embeddings: {e}")
                raise RuntimeError(f"Couldn't initialize Embeddings model '{embed_model}'") from e
        else:
            log.warning(f"Embeddings '{embed_model}' initialized without connection verification.")

        # Initialize Qdrant vector store with retry logic
        max_retries = 10
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try to connect to existing collection first
                try:
                    self.db = Qdrant(
                        url=self.qdrant_url,
                        collection_name=self.collection_name,
                        embeddings=self.embeddings,
                        prefer_grpc=False,  # Use HTTP for better compatibility
                    )
                    log.info(f"Connected to existing Qdrant collection '{self.collection_name}'.")
                except Exception as e:
                    # Collection doesn't exist, need to create it
                    log.info(f"Collection doesn't exist, creating new one...")
                    dummy_doc = Document(
                        page_content="Hello World!",
                        metadata={"user_id": "public", 'source': "test document"}
                    )
                    self.db = Qdrant.from_documents(
                        [dummy_doc],
                        embedding=self.embeddings,
                        url=self.qdrant_url,
                        collection_name=self.collection_name,
                        prefer_grpc=False,
                    )
                    log.info(f"Created new Qdrant collection '{self.collection_name}' with dummy document.")
                break
            except Exception as e:
                last_error = e
                log.warning(f"Attempt {attempt + 1}/{max_retries} failed to initialize Qdrant: {e}")
                if attempt < max_retries - 1:
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        else:
            log.error(f"Failed to initialize Qdrant after {max_retries} attempts. Last error: {last_error}")
            raise RuntimeError(f"Could not initialize Qdrant after {max_retries} attempts") from last_error

        # Create configurable retriever for runtime filtering
        retriever = self.db.as_retriever()
        configurable_retriever = retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )
        self.retriever = configurable_retriever
        log.info(f"Created configurable retriever with k={retriever_num_docs}.")

    def get_embeddings(self) -> Embeddings:
        log.info("Returning the Embeddings model instance.")
        return self.embeddings

    def get_vector_store(self) -> VectorStore:
        log.info("Returning the Qdrant vector store instance.")
        return self.db

    def get_retriever(self) -> VectorStoreRetriever:
        log.info("Returning the retriever for similarity search.")
        return self.retriever # type: ignore[return-value]
    def save_db_to_disk(self) -> bool:
        """No-op for Qdrant.

        Qdrant is a remote, persisted vector database. This method is kept for
        compatibility (some older workflows expect a save function) but does
        nothing and returns True.
        """
        log.info("Qdrant is remote; save_db_to_disk is a no-op and returns True.")
        return True
