"""Hybrid Retrieval Module with Qdrant's Native DBSF, SPLADE Sparse Vectors, and Voyage AI Reranking

This module implements hybrid retrieval using Qdrant's native capabilities:
1. Dense Vector Search via Qdrant's Query API
2. SPLADE Sparse Vector Generation (FastEmbed)
3. DBSF (Density-Biased Similarity Fusion) - Qdrant native (v1.11.0+)
4. MMR (Maximal Marginal Relevance) - Qdrant native (v1.15.0+)
5. Voyage AI Rerank 2.5 for cross-encoder re-ranking
6. RRF (Reciprocal Rank Fusion) as alternative fusion method

References:
- Qdrant Hybrid Queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
- SPLADE with FastEmbed: https://qdrant.tech/documentation/fastembed/fastembed-splade/
- DBSF Paper: https://arxiv.org/abs/2311.03099
- Voyage AI: https://docs.voyageai.com/reference/rerank-2-5
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import numpy as np

# Add parent directories to path for logger import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from logger import get_logger
log = get_logger(name="hybrid_retriever")


class HybridRetriever:
    """Leverages Qdrant's native hybrid retrieval with SPLADE sparse vectors and Voyage AI reranking."""
    
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        embeddings: Embeddings,
        voyage_api_key: Optional[str] = None,
        k: int = 5,
        use_fusion: str = "dbsf",  # "dbsf" or "rrf"
        use_mmr: bool = True,
        use_reranking: bool = True,
        use_splade: bool = True,
        mmr_diversity: float = 0.5
    ):
        """
        Initialize hybrid retriever with SPLADE sparse vectors and Qdrant native capabilities.
        
        Args:
            qdrant_client: Qdrant client instance (from VectorDB)
            collection_name: Qdrant collection name
            embeddings: Embeddings model (dense vectors)
            voyage_api_key: Voyage AI API key for reranking (from env if not provided)
            k: Number of final results to return
            use_fusion: Fusion method - "dbsf" (Density-Biased) or "rrf" (Reciprocal Rank)
            use_mmr: Whether to use MMR for diversity (Qdrant v1.15.0+)
            use_reranking: Whether to use Voyage AI reranking
            use_splade: Whether to generate sparse vectors with SPLADE
            mmr_diversity: Diversity parameter for MMR (0.0 = relevance, 1.0 = diversity)
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.k = k
        self.use_fusion = use_fusion.lower()
        self.use_mmr = use_mmr
        self.use_reranking = use_reranking
        self.use_splade = use_splade
        self.mmr_diversity = mmr_diversity
        self.sparse_model = None
        
        if self.use_fusion not in ["dbsf", "rrf"]:
            log.warning(f"Unknown fusion method '{use_fusion}', defaulting to 'dbsf'")
            self.use_fusion = "dbsf"
        
        # Initialize SPLADE sparse embeddings
        if self.use_splade:
            self._initialize_splade()
        
        # Initialize Voyage AI if reranking is enabled
        self.voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if self.use_reranking and not self.voyage_api_key:
            log.warning("Voyage AI API key not found. Reranking will be disabled.")
            self.use_reranking = False
        
        if self.use_reranking:
            try:
                import voyageai
                self.voyage_client = voyageai.Client(api_key=self.voyage_api_key)
                log.info("Voyage AI client initialized for reranking.")
            except ImportError:
                log.warning("voyageai package not found. Reranking disabled.")
                self.use_reranking = False
            except Exception as e:
                log.warning(f"Failed to initialize Voyage AI: {e}")
                self.use_reranking = False
        
        log.info(
            f"HybridRetriever initialized: fusion={self.use_fusion}, "
            f"SPLADE={self.use_splade}, MMR={self.use_mmr}, reranking={self.use_reranking}, k={k}"
        )
    
    def _initialize_splade(self):
        """Initialize SPLADE sparse embedding model from FastEmbed."""
        try:
            from fastembed import SparseTextEmbedding
            
            model_name = "prithivida/Splade_PP_en_v1"
            self.sparse_model = SparseTextEmbedding(model_name=model_name)
            log.info(f"SPLADE model initialized: {model_name}")
        except ImportError:
            log.warning("fastembed package not found. SPLADE sparse vectors will be disabled.")
            self.use_splade = False
            self.sparse_model = None
        except Exception as e:
            log.error(f"Failed to initialize SPLADE: {e}")
            self.use_splade = False
            self.sparse_model = None
    
    def _embed_query_sparse(self, query: str) -> Optional[Dict[int, float]]:
        """
        Generate sparse embedding for query using SPLADE.
        
        Returns:
            Dictionary mapping token indices to weights, or None if sparse embedding fails
        """
        if not self.use_splade or self.sparse_model is None:
            return None
        
        try:
            # Generate sparse embedding
            sparse_embeddings = list(self.sparse_model.embed(query))
            if not sparse_embeddings:
                log.warning("SPLADE returned empty sparse embedding")
                return None
            
            sparse_emb = sparse_embeddings[0]
            # Convert to dictionary {index: weight}
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_emb.indices, sparse_emb.values)}
            log.info(f"SPLADE sparse embedding generated: {len(sparse_dict)} tokens")
            return sparse_dict
        except Exception as e:
            log.warning(f"SPLADE sparse embedding failed: {e}")
            return None
    
    def _embed_documents_sparse(self, docs: List[str], batch_size: int = 6) -> Optional[List[Dict[int, float]]]:
        """
        Generate sparse embeddings for documents using SPLADE.
        
        Args:
            docs: List of document texts
            batch_size: Batch size for processing
        
        Returns:
            List of sparse embeddings (dict mapping token indices to weights), or None if fails
        """
        if not self.use_splade or self.sparse_model is None:
            return None
        
        try:
            sparse_embeddings = list(self.sparse_model.embed(docs, batch_size=batch_size))
            sparse_dicts = [
                {int(idx): float(val) for idx, val in zip(sparse_emb.indices, sparse_emb.values)}
                for sparse_emb in sparse_embeddings
            ]
            log.info(f"SPLADE sparse embeddings generated for {len(sparse_dicts)} documents")
            return sparse_dicts
        except Exception as e:
            log.warning(f"SPLADE batch embedding failed: {e}")
            return None
    
    def retrieve_hybrid(
        self, 
        query: str,
        dense_limit: int = 20,
        candidates_limit: int = 100
    ) -> List[Document]:
        """
        Perform hybrid retrieval using Qdrant's native Query API with SPLADE sparse vectors.
        
        Steps:
        1. Embed query to dense vector
        2. Generate sparse vector with SPLADE
        3. Execute Qdrant Query API with:
           - Dense vector prefetch
           - Sparse vector prefetch
           - DBSF/RRF fusion
           - Optional MMR for diversity
        4. Optional: Re-rank with Voyage AI
        
        Args:
            query: User query string
            dense_limit: Number of dense results (before fusion)
            candidates_limit: Candidate limit for MMR pre-filtering
        
        Returns:
            List of re-ranked and diverse documents
        """
        log.info(f"Starting hybrid retrieval for query: '{query[:80]}'")
        
        # Step 1: Get query embeddings (both dense and sparse)
        try:
            query_embedding = self.embeddings.embed_query(query)
            log.info(f"Query dense embedding generated: {len(query_embedding)} dimensions")
        except Exception as e:
            log.error(f"Failed to embed query (dense): {e}")
            return []
        
        # Get sparse embedding
        sparse_query_embedding = None
        if self.use_splade:
            sparse_query_embedding = self._embed_query_sparse(query)
            if sparse_query_embedding:
                log.info(f"Query sparse embedding generated: {len(sparse_query_embedding)} tokens")
        
        # Step 2: Build and execute Qdrant Query API request
        try:
            docs = self._execute_qdrant_query(query_embedding, sparse_query_embedding, dense_limit, candidates_limit)
            log.info(f"Qdrant query returned {len(docs)} documents")
        except Exception as e:
            log.error(f"Qdrant query failed: {e}")
            # Fallback to simple vector search
            log.info("Falling back to simple vector search...")
            docs = self._fallback_vector_search(query_embedding)
        
        # Step 3: Apply MMR if enabled (and not already done in Qdrant)
        if self.use_mmr and not hasattr(self, '_mmr_done_in_qdrant'):
            docs = self._apply_mmr(docs, query_embedding)
            log.info(f"MMR applied: {len(docs)} documents")
        
        # Step 4: Re-rank with Voyage AI
        final_docs = self._rerank_with_voyage(query, docs)
        log.info(f"Final hybrid retrieval returned {len(final_docs)} documents")
        
        return final_docs
    
    def _execute_qdrant_query(
        self, 
        query_embedding: List[float],
        sparse_query_embedding: Optional[Dict[int, float]],
        dense_limit: int,
        candidates_limit: int
    ) -> List[Document]:
        """
        Execute Qdrant's Query API with DBSF/RRF fusion combining dense and sparse vectors.
        
        Uses Qdrant native hybrid retrieval capabilities:
        - Prefetch: Dense vector search + Sparse vector search (SPLADE)
        - Query: DBSF or RRF fusion
        - Optional: MMR for diversity
        
        Reference: https://qdrant.tech/documentation/concepts/hybrid-queries/
                  https://qdrant.tech/documentation/fastembed/fastembed-splade/
        """
        log.info(f"Executing Qdrant Query API with {self.use_fusion.upper()} fusion (dense + sparse)")
        
        try:
            # Build query request based on Qdrant Query API with hybrid support
            prefetch_list = [
                {
                    "query": query_embedding,  # Dense vector
                    "limit": dense_limit,
                }
            ]
            
            # Add sparse vector prefetch if available
            if sparse_query_embedding and self.use_splade:
                log.info(f"Adding sparse prefetch with {len(sparse_query_embedding)} tokens")
                prefetch_list.append({
                    "query": sparse_query_embedding,  # Sparse vector as dict
                    "limit": dense_limit,
                })
            
            query_request = {
                "prefetch": prefetch_list,
                "limit": self.k
            }
            
            # Add fusion method
            if self.use_fusion == "dbsf":
                query_request["query"] = {"fusion": "dbsf"}
                log.info("Using DBSF fusion (Density-Biased Similarity Fusion)")
            elif self.use_fusion == "rrf":
                query_request["query"] = {"fusion": "rrf"}
                log.info("Using RRF fusion (Reciprocal Rank Fusion)")
            
            # Note: MMR in Qdrant requires v1.15.0+ and may conflict with prefetch-based fusion
            # For now, we use prefetch-based fusion instead of Qdrant native MMR
            # If MMR is needed, it should be applied client-side after retrieval
            
            # Execute query
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                **query_request
            )
            
            # Convert Qdrant results to LangChain Documents
            docs = self._qdrant_results_to_documents(results)
            log.info(f"Qdrant query returned {len(docs)} documents (dense + sparse fusion)")
            return docs
        
        except Exception as e:
            log.error(f"Qdrant Query API failed: {e}")
            log.error(f"Exception type: {type(e).__name__}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
            # Qdrant version might not support Query API or sparse vectors
            log.info("Falling back to search API...")
            return self._fallback_vector_search(query_embedding)
    
    def _fallback_vector_search(self, query_embedding: List[float]) -> List[Document]:
        """
        Fallback: Simple vector search using Qdrant search API.
        Used if Query API is not available or fails.
        """
        log.info("Using fallback vector search...")
        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.k * 2  # Get more for potential filtering
            )
            docs = self._qdrant_results_to_documents(results)
            return docs
        except Exception as e:
            log.error(f"Fallback search also failed: {e}")
            return []
    
    def _qdrant_results_to_documents(self, results) -> List[Document]:
        """Convert Qdrant search results to LangChain Documents."""
        docs = []
        try:
            for result in results:
                # Handle both scored_point and point objects
                if hasattr(result, 'payload'):
                    payload = result.payload
                elif hasattr(result, 'point') and hasattr(result.point, 'payload'):
                    payload = result.point.payload
                else:
                    continue
                
                # Extract page content
                page_content = payload.get('page_content', 
                                          payload.get('content', 
                                                     str(payload)))
                
                # Extract metadata
                metadata = {k: v for k, v in payload.items() 
                           if k not in ['page_content', 'content']}
                if hasattr(result, 'score'):
                    metadata['score'] = result.score
                
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)
        except Exception as e:
            log.warning(f"Error converting Qdrant results: {e}")
        
        return docs
    
    def _apply_mmr(
        self, 
        docs: List[Document], 
        query_embedding: List[float]
    ) -> List[Document]:
        """
        Apply client-side MMR as fallback if Qdrant MMR is not available.
        """
        if len(docs) <= self.k:
            return docs
        
        log.info(f"Applying client-side MMR: {len(docs)} docs -> {self.k} docs")
        
        try:
            # Get embeddings for all docs
            doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
            doc_embeddings = np.array(doc_embeddings)
            query_emb = np.array(query_embedding)
            
            selected_indices = [0]
            selected_embeddings = [doc_embeddings[0]]
            
            while len(selected_indices) < min(self.k, len(docs)):
                best_idx = -1
                best_score = -float('inf')
                
                for i in range(len(docs)):
                    if i in selected_indices:
                        continue
                    
                    # Relevance: similarity to query
                    relevance = float(np.dot(doc_embeddings[i], query_emb) / 
                                    (np.linalg.norm(doc_embeddings[i]) * np.linalg.norm(query_emb) + 1e-8))
                    
                    # Redundancy: min similarity to already selected
                    min_redundancy = min([float(np.dot(doc_embeddings[i], emb) / 
                                               (np.linalg.norm(doc_embeddings[i]) * np.linalg.norm(emb) + 1e-8))
                                         for emb in selected_embeddings])
                    
                    # MMR: relevance - lambda * redundancy
                    mmr_score = relevance - self.mmr_diversity * min_redundancy
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    selected_embeddings.append(doc_embeddings[best_idx])
            
            result = [docs[i] for i in selected_indices[:self.k]]
            log.info(f"MMR selected {len(result)} diverse documents")
            return result
        
        except Exception as e:
            log.warning(f"MMR failed, returning top-k: {e}")
            return docs[:self.k]
    
    def _rerank_with_voyage(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Re-rank documents using Voyage AI Rerank 2.5.
        """
        if not self.use_reranking or len(docs) == 0:
            return docs[:self.k]
        
        try:
            log.info(f"Re-ranking {len(docs)} documents with Voyage AI...")
            
            # Prepare documents for reranking
            doc_texts = [doc.page_content for doc in docs]
            
            # Call Voyage Rerank API
            reranked = self.voyage_client.rerank(
                query=query,
                documents=doc_texts,
                model="rerank-2.5",
                top_k=self.k
            )
            
            # Map reranked results back to documents
            reranked_docs = [docs[result.index] for result in reranked.results]
            log.info(f"Voyage reranking completed: {len(reranked_docs)} documents returned")
            
            return reranked_docs
        
        except Exception as e:
            log.error(f"Voyage reranking failed: {e}. Returning original docs.")
            return docs[:self.k]
