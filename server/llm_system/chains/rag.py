from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory, Runnable
from langchain_core.documents import Document

from .prompts import template_chat as chat_prompt
from .prompts import template_summarize as summary_prompt

from typing import Callable, List, Any
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel

from logger import get_logger
log = get_logger(name="chains_rag")


class HybridRetrieverWrapper(Runnable):
    """Wrapper to make HybridRetriever compatible with LangChain chains."""
    
    def __init__(self, hybrid_retriever):
        self.hybrid_retriever = hybrid_retriever
    
    def invoke(self, input_data: str, config=None, **kwargs) -> List[Document]:
        """Retrieve documents using hybrid retrieval."""
        if isinstance(input_data, dict):
            query = input_data.get('input', str(input_data))
        else:
            query = str(input_data)
        
        log.info(f"HybridRetrieverWrapper.invoke called with query: '{query[:80]}'")
        docs = self.hybrid_retriever.retrieve_hybrid(query)
        log.info(f"HybridRetrieverWrapper returned {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):
            log.info(f"  Doc {i+1}: {doc.page_content[:100]}...")
        return docs
    
    def batch(self, inputs: List[str], config=None, **kwargs) -> List[List[Document]]:
        """Batch retrieval."""
        return [self.invoke(inp, config) for inp in inputs]
    
    def stream(self, input_data: str, config=None, **kwargs):
        """Stream retrieval results."""
        yield self.invoke(input_data, config)



def build_rag_chain(
        llm_chat: BaseChatModel, llm_summary: BaseChatModel,
        retriever: VectorStoreRetriever, get_history_fn: Callable,
        use_hybrid: bool = False):
    """Builds a Conversational RAG (Retrieval-Augmented Generation) chain.

    Args:
        llm_chat (BaseChatModel): The LLM model for generating chat responses.
        llm_summary (BaseChatModel): The LLM model for summarizing chat history.
        retriever (VectorStoreRetriever): The retriever to fetch relevant documents.
        get_history_fn (Callable): Function to retrieve chat history for a session.
        use_hybrid (bool): Whether to use hybrid retrieval (SPLADE + DBSF + Voyage). Defaults to False.

    Returns:
        RunnableWithMessageHistory: A runnable chain that processes user input and chat history
        to provide a final answer based on retrieved documents and chat context.
    """

    log.info(f"Building the Conversational RAG Chain (hybrid={use_hybrid})...")

    # If using hybrid retriever, wrap it
    if use_hybrid and hasattr(retriever, 'retrieve_hybrid'):
        log.info("Using HybridRetrieverWrapper for advanced retrieval")
        retriever = HybridRetrieverWrapper(retriever)

    # Chain to summarize the history and retrieve relevant documents
    # 3 User Input + Chat History > Summarizer Template > Standalone Que > Get Docs
    retriever_chain = create_history_aware_retriever(llm_summary, retriever, summary_prompt)
    log.info("Created the retriever chain with summarization.")
    log.info(f"Retriever chain type: {type(retriever_chain)}")

    # Chain to combine the retrieved documents and get the final answer
    # 4 Multiple Docs > Combine All > Chat Template > Final Output
    qa_chain = create_stuff_documents_chain(llm=llm_chat, prompt=chat_prompt)
    log.info("Created the QA chain with chat template.")
    log.info(f"QA chain type: {type(qa_chain)}")
    log.info(f"Chat prompt input variables: {chat_prompt.input_variables}")

    # Main RAG Chain:
    # 2 Input + Chat History > [ `Summarizer Template` > `Get Docs` ] > [ `Combine` > `Chat Template` ] > Output
    rag_chain = create_retrieval_chain(retriever_chain, qa_chain)
    log.info("Created the main RAG chain.")
    log.info(f"RAG chain type: {type(rag_chain)}")

    log.info("Returning the final Conversational RAG Chain w history.")
    # 1 Final Conversational RAG Chain:
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
