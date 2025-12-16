"""
Hybrid Retrieval Example and Testing Script

This script demonstrates how to:
1. Initialize the hybrid retriever
2. Use it for document retrieval
3. Benchmark against baseline
4. Test individual components
"""

import asyncio
import time
from typing import List
from langchain_core.documents import Document


def example_basic_usage():
    """Basic example: Initialize and use hybrid retriever."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Hybrid Retrieval")
    print("="*60 + "\n")
    
    from llm_system.core.database import VectorDB
    
    # Initialize vector database
    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        retriever_num_docs=5,
        verify_connection=False
    )
    
    # Get hybrid retriever (all components enabled)
    hybrid_retriever = vector_db.get_hybrid_retriever(
        use_mmr=True,
        use_reranking=True  # Requires VOYAGE_API_KEY env var
    )
    
    # Example queries
    queries = [
        "How to handle errors in production?",
        "Best practices for async programming",
        "Security considerations for APIs"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        try:
            docs = hybrid_retriever.retrieve_hybrid(query)
            
            print(f"Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs, 1):
                content_preview = doc.page_content[:80].replace("\n", " ")
                print(f"  {i}. {content_preview}...")
                print(f"     Source: {doc.metadata.get('source', 'unknown')}")
        
        except Exception as e:
            print(f"Error: {e}")


def example_component_comparison():
    """Compare vector-only vs hybrid retrieval."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Component Comparison")
    print("="*60 + "\n")
    
    from llm_system.core.database import VectorDB
    
    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        retriever_num_docs=3
    )
    
    query = "What are the key features of the system?"
    
    # Method 1: Vector-only
    print("\n1. VECTOR-ONLY RETRIEVAL")
    print("-" * 40)
    retriever_vector = vector_db.get_retriever()
    docs_vector = retriever_vector.invoke(query, k=3)
    print(f"Retrieved {len(docs_vector)} documents")
    for i, doc in enumerate(docs_vector, 1):
        print(f"  {i}. {doc.page_content[:60]}...")
    
    # Method 2: Hybrid with all components
    print("\n2. HYBRID RETRIEVAL (Vector + Sparse + MMR + Voyage)")
    print("-" * 40)
    hybrid_retriever = vector_db.get_hybrid_retriever(
        use_mmr=True,
        use_reranking=True
    )
    docs_hybrid = hybrid_retriever.retrieve_hybrid(query)
    print(f"Retrieved {len(docs_hybrid)} documents")
    for i, doc in enumerate(docs_hybrid, 1):
        print(f"  {i}. {doc.page_content[:60]}...")
    
    # Analysis
    print("\n3. COMPARISON")
    print("-" * 40)
    print(f"Vector-only: {len(docs_vector)} docs")
    print(f"Hybrid: {len(docs_hybrid)} docs")
    print(f"Common: {len(set(d.page_content for d in docs_vector) & set(d.page_content for d in docs_hybrid))} docs")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration")
    print("="*60 + "\n")
    
    from llm_system.core.database import VectorDB
    
    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        retriever_num_docs=5
    )
    
    configurations = [
        ("Vector-only", False, False),
        ("Vector + Sparse", True, False),
        ("Vector + Sparse + MMR", True, False),  # with use_mmr=True in retriever
        ("Full pipeline", True, True),
    ]
    
    query = "Tell me about error handling strategies"
    
    for config_name, use_fusion, use_reranking in configurations:
        print(f"\nConfiguration: {config_name}")
        print("-" * 40)
        
        if config_name == "Vector-only":
            retriever = vector_db.get_retriever()
            docs = retriever.invoke(query, k=5)
        else:
            hybrid = vector_db.get_hybrid_retriever(
                use_mmr=(config_name != "Vector + Sparse"),
                use_reranking=use_reranking
            )
            docs = hybrid.retrieve_hybrid(query)
        
        print(f"Documents returned: {len(docs)}")
        if docs:
            print(f"Top result: {docs[0].page_content[:70]}...")


def example_with_rag_chain():
    """Example integrating hybrid retriever with RAG chain."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Hybrid Retriever in RAG Chain")
    print("="*60 + "\n")
    
    from llm_system.core.database import VectorDB
    from llm_system.core.llm import get_llm
    from llm_system.core.history import HistoryStore
    from llm_system.chains.rag import build_rag_chain
    from llm_system import config
    
    # Initialize components
    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        retriever_num_docs=3
    )
    
    # Get LLMs
    llm_chat = get_llm(
        model_name=config.LLM_CHAT_MODEL_NAME,
        temperature=config.LLM_CHAT_TEMPERATURE
    )
    
    llm_summary = get_llm(
        model_name=config.LLM_SUMMARY_MODEL_NAME,
        temperature=config.LLM_SUMMARY_TEMPERATURE
    )
    
    # Get hybrid retriever
    hybrid_retriever = vector_db.get_hybrid_retriever(
        use_mmr=True,
        use_reranking=True
    )
    
    # Initialize history
    history_store = HistoryStore()
    
    # Build RAG chain with hybrid retriever
    rag_chain = build_rag_chain(
        llm_chat=llm_chat,
        llm_summary=llm_summary,
        retriever=hybrid_retriever,
        get_history_fn=history_store.get_session_history,
        use_hybrid=True
    )
    
    print("RAG Chain built with hybrid retriever!")
    print(f"Chain type: {type(rag_chain)}")
    print(f"Chat LLM: {config.LLM_CHAT_MODEL_NAME}")
    print(f"Summary LLM: {config.LLM_SUMMARY_MODEL_NAME}")
    print(f"Embeddings: {config.EMB_MODEL_NAME}")
    print(f"\nHybrid retrieval enabled with:")
    print(f"  - SPLADE semantic search")
    print(f"  - DBSF fusion")
    print(f"  - MMR diversity")
    print(f"  - Voyage AI reranking")


def example_performance_benchmark():
    """Benchmark hybrid retriever performance."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Performance Benchmark")
    print("="*60 + "\n")
    
    from llm_system.core.database import VectorDB
    
    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        retriever_num_docs=3
    )
    
    test_queries = [
        "error handling in production",
        "async programming patterns",
        "API security best practices",
        "database optimization",
        "caching strategies"
    ]
    
    results = {
        "vector_only": {"times": [], "doc_counts": []},
        "hybrid": {"times": [], "doc_counts": []}
    }
    
    print("Running benchmark on {} queries...\n".format(len(test_queries)))
    
    # Vector-only benchmark
    print("1. VECTOR-ONLY RETRIEVAL")
    print("-" * 40)
    retriever = vector_db.get_retriever()
    
    for query in test_queries:
        start = time.time()
        docs = retriever.invoke(query, k=3)
        elapsed = time.time() - start
        
        results["vector_only"]["times"].append(elapsed)
        results["vector_only"]["doc_counts"].append(len(docs))
        print(f"  {query[:30]:30} | {elapsed*1000:6.1f}ms | {len(docs):2} docs")
    
    # Hybrid benchmark
    print("\n2. HYBRID RETRIEVAL")
    print("-" * 40)
    hybrid_retriever = vector_db.get_hybrid_retriever(
        use_mmr=True,
        use_reranking=False  # Disable Voyage to avoid API calls in benchmark
    )
    
    for query in test_queries:
        start = time.time()
        docs = hybrid_retriever.retrieve_hybrid(query)
        elapsed = time.time() - start
        
        results["hybrid"]["times"].append(elapsed)
        results["hybrid"]["doc_counts"].append(len(docs))
        print(f"  {query[:30]:30} | {elapsed*1000:6.1f}ms | {len(docs):2} docs")
    
    # Summary statistics
    print("\n3. STATISTICS")
    print("-" * 40)
    
    import statistics
    
    for method in ["vector_only", "hybrid"]:
        times = results[method]["times"]
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        avg_docs = statistics.mean(results[method]["doc_counts"])
        
        print(f"\n{method.upper()}:")
        print(f"  Avg time: {avg_time*1000:6.1f}ms")
        print(f"  Min time: {min_time*1000:6.1f}ms")
        print(f"  Max time: {max_time*1000:6.1f}ms")
        print(f"  Avg docs: {avg_docs:4.1f}")
    
    # Comparison
    avg_vector = statistics.mean(results["vector_only"]["times"])
    avg_hybrid = statistics.mean(results["hybrid"]["times"])
    slowdown = avg_hybrid / avg_vector if avg_vector > 0 else 0
    
    print(f"\nHybrid is {slowdown:.1f}x slower than vector-only")


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║     HYBRID RETRIEVAL SYSTEM - EXAMPLES & TESTING       ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Component Comparison", example_component_comparison),
        ("Custom Configuration", example_custom_configuration),
        ("RAG Chain Integration", example_with_rag_chain),
        ("Performance Benchmark", example_performance_benchmark),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun specific example:")
    print("  python hybrid_retriever_examples.py 1")
    print("  python hybrid_retriever_examples.py all")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
