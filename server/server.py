# FastAPI server which will handle all the backend and GenAI aspects of the application
# uvicorn server:app --reload
# Avoid using --reload flag, because, LLMs will keep reloading and system will overheat.

from fastapi import FastAPI, File, UploadFile, Form, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import json
from pydantic import BaseModel
from contextlib import asynccontextmanager

# llm system imports:
from llm_system.core.llm import get_llm, get_output_parser  # Functions
from llm_system.core.llm import get_dummy_response          # Function
from llm_system.core.llm import get_dummy_response_stream   # Function
from llm_system.core.database import VectorDB               # Class
from llm_system.core.history import HistoryStore            # Class
from llm_system.chains.rag import build_rag_chain           # Function
from llm_system import config                               # Constants
from llm_system.core.ingestion import ingest_file           # Function

# Helper Modules:
import pg_db
import files

# Type hinting imports:
from langchain_core.vectorstores import VectorStore as T_VECTOR_STORE
from langchain_core.messages import BaseMessage as T_MESSAGE

import logger
log = logger.get_logger("rag_server")


# Helper function to detect if query is about conversation history using LLM
async def is_conversation_history_question(query: str, llm) -> bool:
    """Use LLM to intelligently detect if the query is asking about previous messages or conversation history."""
    detection_prompt = f"""You are a classifier. Your task is to determine if the user's query is asking about their own PREVIOUS QUESTIONS or CONVERSATION HISTORY.

User query: "{query}"

Answer ONLY with "yes" or "no":
- Answer "yes" if the user is asking about what they said before, their previous question, earlier messages, or the conversation history
- Answer "no" if the user is asking about anything else (documents, topics, facts, etc.)

Common patterns for "yes":
- "what did I ask"
- "my last/first/previous question"
- "what was my 1st/2nd question"
- "earlier in our chat"
- "conversation history"

Always respond with EXACTLY one word: yes or no"""
    
    try:
        response = await llm.ainvoke(detection_prompt)
        
        # Extract content from AIMessage or convert to string
        if hasattr(response, 'content'):
            response_text = response.content.strip().lower()
        else:
            response_text = str(response).strip().lower()
        
        is_history = response_text.startswith("yes")
        log.info(f"[DETECT] Query: '{query}' | LLM Response: '{response_text}' | Detected: {is_history}")
        return is_history
    except Exception as e:
        log.warning(f"[DETECT] Error in is_conversation_history_question: {e}")
        return False


def get_conversation_answer(history_messages: list, query: str) -> str:
    """Generate an answer based on conversation history without using RAG."""
    # Ensure we have properly parsed messages
    parsed_messages = []
    for msg in history_messages:
        if isinstance(msg, dict):
            parsed_messages.append(msg)
        elif isinstance(msg, str):
            try:
                parsed_messages.append(json.loads(msg))
            except:
                pass
    
    # Filter to only human messages
    human_messages = [msg for msg in parsed_messages if isinstance(msg, dict) and msg.get("role") == "human"]
    
    if not human_messages:
        return "I don't have any previous questions in our conversation history."
    
    query_lower = query.lower()
    
    # Try to identify which message they're asking about
    if "last" in query_lower or "most recent" in query_lower or "recent" in query_lower:
        last_question = human_messages[-1].get("content", "")
        return f"Your last question was: \"{last_question}\""
    
    elif "first" in query_lower or "1st" in query_lower or "beginning" in query_lower or "start" in query_lower:
        first_question = human_messages[0].get("content", "")
        return f"Your first question was: \"{first_question}\""
    
    elif "second" in query_lower or "2nd" in query_lower:
        if len(human_messages) > 1:
            second_question = human_messages[1].get("content", "")
            return f"Your second question was: \"{second_question}\""
        else:
            return "You don't have a second question in our conversation."
    
    elif "previous" in query_lower or "before" in query_lower or "prior" in query_lower:
        if len(human_messages) > 1:
            prev_question = human_messages[-2].get("content", "")
            return f"Your previous question was: \"{prev_question}\""
        else:
            return "There are no previous questions before this one."
    
    else:
        # Generic conversation history response - list all questions
        questions = "\n".join([f"• {msg.get('content', '')}" for msg in human_messages])
        return f"Here are all your questions in this conversation:\n{questions}"


# ------------------------------------------------------------------------------
# Constants:
# ------------------------------------------------------------------------------

# UPLOADS_DIR: str = "user_uploads"
OLD_FILE_THRESHOLD: int = 3600 * 1  # 24 hours in seconds
# OLD_FILE_THRESHOLD: int = 20         # 1 min


# ------------------------------------------------------------------------------
# FastAPI Startup:
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the lifespan context manager for startup/shutdown"""

    # [ Startup ]
    log.info("[LifeSpan] Starting the server components.")

    app.state.llm_chat = get_llm(
        model_name=config.LLM_CHAT_MODEL_NAME,
        context_size=config.MAX_CONTENT_SIZE,
        temperature=config.LLM_CHAT_TEMPERATURE,
        verify_connection=config.VERIFY_LLM_CONNECTION
    )

    # app.state.llm_summary = get_llm(...)
    app.state.llm_summary = app.state.llm_chat

    app.state.output_parser = get_output_parser()
    app.state.vector_db = VectorDB(
        embed_model=config.EMB_MODEL_NAME,
        retriever_num_docs=config.DOCS_NUM_COUNT,
        verify_connection=config.VERIFY_EMB_CONNECTION,
        qdrant_host=config.QDRANT_HOST,
        qdrant_port=config.QDRANT_PORT,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    app.state.history_store = HistoryStore()

    # Initialize retriever (standard or hybrid)
    if config.USE_HYBRID_RETRIEVAL:
        log.info("Initializing HYBRID retrieval (SPLADE sparse + dense DBSF + Voyage Reranking)...")
        retriever = app.state.vector_db.get_hybrid_retriever(
            use_splade=config.USE_SPLADE_SPARSE,
            use_mmr=config.USE_MMR_DIVERSITY,
            use_reranking=config.USE_VOYAGE_RERANKING
        )
    else:
        log.info("Initializing standard VECTOR-ONLY retrieval...")
        retriever = app.state.vector_db.get_retriever()

    app.state.rag_chain = build_rag_chain(
        llm_chat=app.state.llm_chat,
        llm_summary=app.state.llm_summary,
        retriever=retriever,
        get_history_fn=app.state.history_store.get_session_history,
        use_hybrid=config.USE_HYBRID_RETRIEVAL,
    )

    log.info("[LifeSpan] All LLM components initialized.")

    # pg_db.delete_database()
    pg_db.create_tables()

    # Files
    files.check_create_uploads_folder()
    files.delete_empty_user_folders()

    # [ Lifespan ]
    yield

    # [ Shutdown ]
    log.info("[LifeSpan] Shutting down LLM server...")
    # Add any cleanup part here
    # Like saving vector DB, or shutting down subprocesses


# Make one FastAPI app instance with the lifespan context manager
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:5500",
        # "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ------------------------------------------------------------------------------
# Basic API Endpoints:
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {
        "message": "LLM RAG Server is running!",
        "further": "Proceed to code ur application :)",
        "thought": "You really are not supposed to be reading this waste of time, but if you are, then you are a curious person. I like that! 😄",
    }


# Define data model for chat request
class BasicChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple")
async def simple(request: Request, chat_request: BasicChatRequest):
    """Endpoint to handle ont time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return JSON with `{"response": "", "session_id": ""}` structure.
    """

    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    try:
        query = chat_request.query
        dummy = chat_request.dummy
        log.info(f"/simple Requested by '{session_id}'")

        if dummy:
            log.info(f"/simple Dummy response returned for '{session_id}'")
            return get_dummy_response()

        else:
            result = await llm.ainvoke(input=query)

            log.info(f"/simple Response generated for '{session_id}'.")
            return {"response": result, "session_id": session_id}

    except Exception as e:

        log.exception(f"/simple Error {e} for '{session_id}'")
        return JSONResponse(status_code=500, content={"error": str(e)})


# Make one streaming endpoint for the Simple LLM response:
class StreamChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple/stream")
async def chat_stream(request: Request, chat_request: StreamChatRequest):
    """Endpoint to handle streaming responses for one time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return NDJSON with types "metadata", "content", or "error".
    """
    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    async def token_streamer():
        try:
            dummy = chat_request.dummy
            s = 'dummy' if dummy else 'real'
            log.info(f"/simple/stream {s} response requested by '{session_id}'")

            # Start be sending meta data first.
            yield json.dumps({
                "type": "metadata",
                "data": {"session_id": session_id}
            }) + "\n"
            # NDJSON (newline-delimited JSON) - Frontend will merge full response my splitting this

            #  Then send the actual response content:
            if dummy:
                # If dummy is True, stream dummy response
                resp = get_dummy_response_stream(
                    batch_tokens=config.BATCH_TOKEN_PS,
                    token_rate=config.TOKENS_PER_SEC
                )
                for chunk in resp:
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            else:
                async for chunk in llm.astream(chat_request.query):
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            # In the end, you can send some "Done" etc if u need some conditional logic
            # Server will auto send EOF to mark end of generator response.
            # yield json.dumps({
            #     "type": "end",
            #     "data": "done"
            # }) + "\n"
            log.info(f"/simple/stream Streaming completed for '{session_id}'")

        except Exception as e:
            log.exception(f"/simple/stream Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"

    # Return a StreamingResponse with the token streamer generator (basically enable streaming)
    return StreamingResponse(token_streamer(), media_type="text/plain")


# ------------------------------------------------------------------------------
# Initialization End-points:
# ------------------------------------------------------------------------------

# Helper function to delete old files and embeddings:
def delete_old_files(user_id: str, time: int = OLD_FILE_THRESHOLD):
    """Function to delete old files and embeddings older than the specified time."""
    log.info(
        f"/delete Deleting old files and embeddings for user '{user_id}' older than {time} seconds")

    # Delete old files
    old_files = pg_db.get_old_files(user_id=user_id, time=time)
    log.info(f"/delete Got old_files result: {old_files}")
    if old_files['files']:
        log.info(f"/delete Removing old files for user '{user_id}': {old_files['files']}")

        for file in old_files['files']:
            status = files.delete_file(user_id=user_id, file_name=file)
            log.info(f"/delete Deleted file '{file}': {status}")
            if status:
                file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=file)
                log.info(f"/delete Got file_id {file_id} for file '{file}'")
                pg_db.mark_file_removed(user_id=user_id, file_id=file_id)
                log.info(f"/delete Marked file '{file}' (ID {file_id}) as removed")

    # Delete old embeddings
    if old_files['embeddings']:
        log.info(f"/delete Removing old embeddings for user '{user_id}'")
        vs: VectorDB = app.state.vector_db
        db: T_VECTOR_STORE = vs.get_vector_store()
        resp = db.delete(old_files['embeddings'])

        # Qdrant auto-persists, no need to save to disk

        if resp == True:
            pg_db.mark_embeddings_removed(vector_ids=old_files['embeddings'])
            log.info(f"/delete Old embeddings removed for user '{user_id}'")
        else:
            log.error(f"/delete Failed to remove old embeddings for user '{user_id}': {resp}")
    else:
        log.info(f"/delete No old files found for user '{user_id}'")


# First end-point to call on client initialization:
class LoginRequest(BaseModel):
    login_id: str
    password: str


@app.post("/login")
async def login(request: Request, login_request: LoginRequest):
    """Endpoint to handle user login.
    + Client sends login_id and password for login
    + Based on it, server authenticates user.
    + user_id is retrieved (for now, it is same as login_id)
    + ? Based on user_id chat history of user is retrieved n returned.

    * Folder is created for user_id, older files are removed
    * Later on, will add on scheduled job to delete old items and will remove the old file deletion logic from here.

    - Post request expects JSON `{"login_id": "", "password": ""}` structure.
    - Return JSON with `{"user_id": "user_id", "chat_history": [user chat history]}` structure.
    """

    login_id = login_request.login_id.strip()
    password = login_request.password.strip()
    log.info(f"/login Requested by '{login_id}'")

    # Check if the user exists in the database
    status, msg = pg_db.authenticate_user(user_id=login_id, password=password)
    if status:
        user_id = login_id
        # Check if folder exists in UPLOADS_DIR with user_id
        files.create_user_uploads_folder(user_id=user_id)
        # Skip old file deletion to avoid Qdrant format errors
        # delete_old_files(user_id=user_id, time=OLD_FILE_THRESHOLD)
        return JSONResponse(content={"user_id": user_id, "name": msg}, status_code=200)
    else:
        return JSONResponse(content={"error": msg}, status_code=401)

    # # For now, we will just return a dummy user_id
    # # In future, can implement actual user authentication and return a real user_id
    # user_id = login_id
    # log.info(f"/login requested by '{user_id}'")

    # # Check if folder exists in UPLOADS_DIR with user_id
    # files.create_user_uploads_folder(user_id=user_id)

    # # Old any older data if exists (older than 24 hours)
    # delete_old_files(user_id=user_id, time=OLD_FILE_THRESHOLD)

    # # Get the chat history for the user_id
    # hs: HistoryStore = request.app.state.history_store
    # history = hs.get_session_history(session_id=user_id)
    # if not history:
    #     log.info(f"/login No history found for user '{user_id}'")
    # else:
    #     log.info(f"/login History found for user '{user_id}' with {len(history.messages)} messages")

    # return {"user_id": user_id, "chat_history": history.messages}


# endpoint for user registration:
class RegisterRequest(BaseModel):
    name: str
    user_id: str
    password: str


@app.post("/register")
async def register(request: Request, register_request: RegisterRequest):
    """Endpoint to handle user registration.
    - Post request expects JSON `{"user_name": "Full Name", "user_id": "any_u_id", "password": "raw_pw"}` structure.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """

    name = register_request.name.strip()
    user_id = register_request.user_id.strip()
    password = register_request.password.strip()
    log.info(f"/register Requested by {name} with '{user_id}'")
    print(f"Name: {name}, UserID: {user_id}, Password: {password}")

    # Check if the user already exists
    status = pg_db.check_user_exists(user_id=user_id)
    if status:
        log.error(f"/register UserID '{user_id}' already exists.")
        return JSONResponse(content={"error": "User already exists"}, status_code=400)

    # If user does not exist, add the user to the database
    status = pg_db.add_user(user_id=user_id, name=name, password=password)
    if status:
        return JSONResponse(content={"status": "success"}, status_code=201)
    else:
        return JSONResponse(content={"error": "Failed to register user"}, status_code=500)


# ------------------------------------------------------------------------------
# Chat History Endpoints:
# ------------------------------------------------------------------------------

# Endpoint to get chat history for user:
@app.post("/chat_history")
async def chat_history(user_id: str = Form(...)):
    """Endpoint to get chat history for user.
    - Post request expects `user_id` as form parameter.
    - Return JSON with `{"chat_history": [user chat history]}` or `{"error": "message"}` structure.
    """
    log.info(f"/chat_history Requested by '{user_id}'")
    hs: HistoryStore = app.state.history_store
    history = hs.get_session_history(session_id=user_id)

    if history:
        messages = []
        for msg in history.messages:
            msg: T_MESSAGE
            if msg.type == "ai":
                messages.append({"role": "assistant", "content": msg.text()})
            elif msg.type == "human":
                messages.append({"role": "human", "content": msg.text()})

        return JSONResponse(content={"chat_history": messages}, status_code=200)
    else:
        return JSONResponse(content={"error": "No chat history found"}, status_code=404)


# Endpoint /clear_chat_history to clear chat history for user:
@app.post("/clear_chat_history")
async def clear_chat_history(user_id: str = Form(...)):
    """Endpoint to clear chat history for user.
    - Post request expects `user_id` as form parameter.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """
    log.info(f"/clear_chat_history Requested by '{user_id}'")
    hs: HistoryStore = app.state.history_store
    status = hs.clear_session_history(session_id=user_id)

    if status:
        return JSONResponse(content={"status": "success"}, status_code=200)
    else:
        return JSONResponse(content={"error": "No history found to clear"}, status_code=404)


# ------------------------------------------------------------------------------
# File handling endpoints:
# ------------------------------------------------------------------------------

# Endpoint to receive file uploads:
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    log.info(f"/upload Received file: {file.filename} from user: {user_id}")
    filename = file.filename if file.filename else "unknown_file"

    status, message = files.save_file(
        user_id=user_id,
        file_value_binary=await file.read(),
        file_name=filename
    )

    if status:
        filename = message
        pg_db.add_file(user_id=user_id, filename=filename)
        return JSONResponse(content={"message": filename}, status_code=200)
    else:
        log.error(f"/upload File upload failed for user {user_id}: {filename}")
        return JSONResponse(content={"error": message}, status_code=500)


# Endpoint to embed the uploaded file:
# takes user_id and file_name as input
class EmbedRequest(BaseModel):
    user_id: str
    file_name: str


@app.post("/embed")
async def embed_file(embed_request: EmbedRequest, request: Request):
    """Endpoint to embed the uploaded file.
    - Post request expects JSON `{"user_id": "", "file_name": ""}` structure.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """
    user_id = embed_request.user_id.strip()
    file_name = embed_request.file_name.strip()

    log.info(f"/embed Requested by '{user_id}' for file '{file_name}'")

    # Call the ingest_file function to process the file
    status, doc_ids, message = ingest_file(
        user_id=user_id,
        file_path=files.get_file_path(user_id=user_id, file_name=file_name),
        vectorstore=request.app.state.vector_db,
        embeddings=request.app.state.vector_db.get_embeddings()
    )

    if status:
        file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=file_name)
        for vid in doc_ids:
            pg_db.add_embedding(file_id=file_id, vector_id=vid)

        log.info(f"/embed Embedding completed for '{user_id}' and file '{file_name}'")
        return JSONResponse(content={"status": "success"}, status_code=200)
    else:
        log.error(f"/embed Embedding failed for '{user_id}' and file '{file_name}': {message}")
        return JSONResponse(content={"error": message}, status_code=500)


# ------------------------------------------------------------------------------
# Data management endpoints:
# ------------------------------------------------------------------------------

# Endpoint /clear_my_files to clear all files uploaded by user:
@app.post("/clear_my_files")
async def clear_my_files(user_id: str = Form(...)):
    """Endpoint to clear all files uploaded by user.
    - Post request expects `user_id` as form parameter.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """

    log.info(f"/clear_my_files Requested by '{user_id}'")
    delete_old_files(user_id=user_id, time=1)
    return JSONResponse(content={"status": "success"}, status_code=200)


# End point to get all the files uploaded by user:
# This will be called first at initialization, and then after each file upload
@app.get("/uploads")
async def get_files(user_id: str = Query(...)):
    """Endpoint to get all the files uploaded by user.
    - Get request expects `user_id` as query parameter.
    - Return JSON with `{"files": ["file1", "file2", ...]}` structure.
    """
    log.info(f"/uploads Requested by '{user_id}'")
    files_list = pg_db.get_user_files(user_id=user_id)
    return {"files": files_list}


# Send pdf iframe based on user and file name:
# params: type=pdf/ppt/txt, user_id, file_name, num_pages
class FileIframeRequest(BaseModel):
    # type: Literal["pdf", "ppt", "txt"]
    user_id: str
    file_name: str
    num_pages: int = 5


@app.post("/iframe")
async def get_file_iframe(file_request: FileIframeRequest):
    """Endpoint to get the iframe for the file.
    - Post request expects JSON `{"user_id": "", "file_name": "", "num_pages": 5}` structure.
    - Return JSON with `{"iframe": "<iframe>...</iframe>"}` structure.
    """

    user_id = file_request.user_id.strip()
    file_name = file_request.file_name.strip()
    num_pages = file_request.num_pages

    log.info(f"/iframe Requested by '{user_id}' for file '{file_name}'")

    # Get the iframe for the requested file
    status, message = files.get_pdf_iframe(
        user_id=user_id,
        file_name=file_name,
        num_pages=num_pages
    )

    if status:
        return JSONResponse(content={"iframe": message}, status_code=200)
    else:
        return JSONResponse(content={"error": message}, status_code=404)


# ------------------------------------------------------------------------------
# RAG Chain Endpoint:
# ------------------------------------------------------------------------------

# Create endpoint for rag:
# input = {
#     query: str,
#     session_id: str,
#     dummy: bool = False
# }
# Output will be streamed in same format as the simple/streaming chat endpoint.


class RagChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/rag")
async def rag(request: Request, chat_request: RagChatRequest):
    """Endpoint to handle RAG (Retrieval-Augmented Generation) queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return NDJSON with types "metadata", "content", "context", or "error".
    """
    rag_chain = request.app.state.rag_chain
    session_id = chat_request.session_id.strip() or "unknown_session"

    async def token_streamer():
        try:
            dummy = chat_request.dummy
            log.info(f"/rag {'dummy' if dummy else 'real'} response requested by '{session_id}'")

            # Start by sending meta data first.
            yield json.dumps({
                "type": "metadata",
                "data": {"session_id": session_id}
            }) + "\n"

            if dummy:
                # If dummy is True, stream dummy response
                resp = get_dummy_response_stream(
                    batch_tokens=config.BATCH_TOKEN_PS,
                    token_rate=config.TOKENS_PER_SEC
                )
                for chunk in resp:
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            else:
                log.info(f"/rag Starting non-dummy response for query: '{chat_request.query[:80]}'")
                
                # Check if this is a conversation history question using LLM-based detection
                # This is more robust than keyword matching and handles spelling variations
                is_history_question = await is_conversation_history_question(chat_request.query, request.app.state.llm_chat)
                log.info(f"/rag Detection result: {is_history_question}")
                
                if is_history_question:
                    print(f"DEBUG: Entering is_history_question=True branch")
                    log.info(f"/rag Detected conversation history question: '{chat_request.query}'")
                    
                    # Get chat history directly from Redis (raw JSON messages)
                    try:
                        print(f"DEBUG: Trying to get Redis messages")
                        from llm_system.core.history import redis_client
                        key = f"chat_history:{session_id}"
                        print(f"DEBUG: Redis key: {key}")
                        raw_messages = redis_client.lrange(key, 0, -1)
                        print(f"DEBUG: Got {len(raw_messages)} messages from Redis")
                        
                        # Parse JSON messages from Redis
                        history_messages = []
                        for msg_bytes in raw_messages:
                            try:
                                msg_dict = json.loads(msg_bytes.decode('utf-8') if isinstance(msg_bytes, bytes) else msg_bytes)
                                history_messages.append(msg_dict)
                            except Exception as parse_err:
                                print(f"DEBUG: Failed to parse message: {parse_err}")
                                pass
                        
                        print(f"DEBUG: Parsed {len(history_messages)} messages")
                        log.info(f"/rag Loaded {len(history_messages)} messages from Redis for {session_id}")
                        
                        # Generate answer directly from conversation history
                        print(f"DEBUG: Calling get_conversation_answer with {len(history_messages)} messages")
                        answer = get_conversation_answer(history_messages, chat_request.query)
                        print(f"DEBUG: Got answer: {answer[:100]}")
                        log.info(f"/rag Generated answer from conversation history: {answer[:80]}...")
                        
                        yield json.dumps({
                            "type": "content",
                            "data": answer
                        }) + "\n"
                    except Exception as e:
                        import traceback
                        print(f"DEBUG: EXCEPTION in conversation history: {e}")
                        print(f"DEBUG: Traceback: {traceback.format_exc()}")
                        log.error(f"/rag Error getting conversation history: {e}")
                        log.error(f"/rag Traceback: {traceback.format_exc()}")
                        yield json.dumps({
                            "type": "error",
                            "data": f"Error retrieving conversation history: {str(e)}"
                        }) + "\n"
                
                else:
                    # Regular RAG chain processing for document-related queries
                    # Search kwargs for the configurable retriever (Qdrant)
                    search_kwargs = {
                        "k": 5,
                    }
                    
                    log.info(f"/rag Processing query: '{chat_request.query[:100]}...' for session '{session_id}'")
                    log.info(f"/rag Search kwargs: {search_kwargs}")

                    async for chunk in rag_chain.astream(
                        input={"input": chat_request.query},
                        config={
                            "configurable": {
                                "session_id": session_id,
                                "search_kwargs": search_kwargs
                            }
                        }
                    ):
                        if await request.is_disconnected():
                            log.warning(f"/rag client disconnected for '{session_id}'")
                            break

                        # there is answer/input/context
                        if "answer" in chunk:
                            log.debug(f"/rag Answer chunk: {chunk['answer'][:100] if isinstance(chunk['answer'], str) else chunk['answer']}")
                            yield json.dumps({
                                "type": "content",
                                "data": chunk["answer"]
                            }) + "\n"

                        elif "context" in chunk:
                            log.info(f"/rag Retrieved {len(chunk['context'])} context documents")
                            for i, document in enumerate(chunk["context"]):
                                log.info(f"/rag   Context Doc {i+1}: {document.page_content[:100]}...")
                                if await request.is_disconnected():
                                    log.warning(f"/rag client disconnected for '{session_id}'")
                                    break

                                # Hide user_id from metadata on UI
                                if "user_id" in document.metadata:
                                    if document.metadata["user_id"] == "public":
                                        document.metadata["isPublicDocument"] = True
                                    else:
                                        document.metadata["isPublicDocument"] = False
                                    document.metadata.pop("user_id")

                                yield json.dumps({
                                    "type": "context",
                                    "data": {
                                        "metadata": document.metadata,
                                        "page_content": document.page_content
                                    }
                                }) + "\n"

            log.info(f"/rag Streaming completed for '{session_id}'")

        except Exception as e:
            log.exception(f"/rag Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"

    return StreamingResponse(token_streamer(), media_type="text/plain")


# ------------------------------------------------------------------------------
# Run the FastAPI server:
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("WARNING: Starting server without explicit uvicorn command. Not recommended for production use.")
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
