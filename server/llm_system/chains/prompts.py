"""Contains the prompt templates for chat and summarization tasks."""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from logger import get_logger
log = get_logger(name="chains_prompts")

# Chat Template:
template_chat = ChatPromptTemplate.from_messages(
    messages=[
        ("system",  (
            "You are a document assistant. Answer questions using ONLY the provided context documents below.\n"
            "Do not use your training knowledge - only extract information from the documents.\n"
            "If the answer is in the context, provide it directly without disclaimers.\n"
            "{context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)


# Summarizer Template:
template_summarize = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "".join([
            "You are an expert at summarizing conversations into standalone prompts.\n"
            "You are given a complete chat history, ending with the user's latest message.\n\n"
            "SPECIAL CASE - If the user is asking about PREVIOUS MESSAGES or CONVERSATION HISTORY:\n"
            "- Detect keywords like: 'my last question', 'what did I ask', 'my first question', 'previous message', etc.\n"
            "- If detected, return EXACTLY what they asked without modification\n"
            "- Do NOT reformulate questions about the conversation itself\n\n"
            "NORMAL CASE - For all other questions:\n"
            "- Understand the entire conversation context.\n"
            "- Identify references in the latest user message that relate to earlier messages.\n"
            "- Create a single clear, concise, and standalone question or prompt.\n"
            "- This final prompt should be fully understandable without needing the prior conversation.\n"
            "- It will be used to retrieve relevant documents.\n\n"
            "EXAMPLES:\n"
            "- User: 'what was my last question?' → Return: 'what was my last question?'\n"
            "- User: 'tell me more about it' (referring to habits from earlier) → Return: 'Tell me more about the good workplace habits'\n"
            "- User: 'my first question' → Return: 'my first question'\n\n"
            "Only return the rewritten standalone prompt (or original if it's about conversation history). No explanations."
        ])),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

log.info("Initialized chat and summarize prompt templates.")
