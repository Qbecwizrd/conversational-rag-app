import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = FastAPI()

# --- CORS Configuration ---
# Allows the frontend (running on a different port) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables & Models ---
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
llm = ChatGroq(groq_api_key=os.getenv('GROQ_API'), model_name='gemma2-9b-it')

# In-memory storage for session data (rag_chains and histories)
# In a production app, you'd use a more persistent store like Redis or a database.
session_store: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    session_id: str
    input: str

class ChatResponse(BaseModel):
    answer: str
    chat_history: List[Dict[str, str]]

# --- Helper Functions ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = {"history": ChatMessageHistory()}
    return session_store[session_id]["history"]

def create_rag_chain_for_session(session_id: str, documents: list):
    """Creates and stores a RAG chain for a given session ID."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Use a persistent directory for ChromaDB specific to the session
    persist_directory = f"./db/{session_id}"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory) # Clear old session data
    os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever()

    # Contextualize prompt
    contextualize_q_system_prompt = (
        'Given a chat history and the latest user question, which might reference '
        'context from the chat history, formulate a standalone question which can be '
        'understood without the chat history. Do not answer the question—just reformulate it if needed.'
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt
    system_prompt = (
        'You are an assistant for question-answering tasks. Use the following pieces of '
        'retrieved context to answer the question. If you don’t know the answer, say that '
        'you don’t know. Use three sentences maximum and keep the answer concise.\n\n{context}'
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda sid: get_session_history(sid),  # Use lambda to pass session_id correctly
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )
    
    # Store the created chain in our session store
    if session_id not in session_store:
        session_store[session_id] = {"history": ChatMessageHistory()}
    session_store[session_id]["rag_chain"] = conversational_rag_chain
    return conversational_rag_chain

# --- API Endpoints ---

@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Endpoint to upload PDFs and initialize a RAG chain for a session."""
    documents = []
    temp_dir = "./temp_pdf"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file in files:
        temp_filepath = os.path.join(temp_dir, file.filename)
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()
        documents.extend(docs)

    shutil.rmtree(temp_dir) # Clean up temp files

    if not documents:
        raise HTTPException(status_code=400, detail="No documents could be loaded from the PDFs.")

    create_rag_chain_for_session(session_id, documents)
    
    return {"status": "success", "session_id": session_id, "message": f"{len(files)} PDF(s) processed."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """Endpoint to handle a chat message."""
    session_id = request.session_id
    if session_id not in session_store or "rag_chain" not in session_store[session_id]:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    conversational_rag_chain = session_store[session_id]["rag_chain"]
    
    response = conversational_rag_chain.invoke(
        {"input": request.input},
        config={"configurable": {"session_id": session_id}}
    )
    
    history = get_session_history(session_id)
    history_dicts = [{"type": msg.type, "content": msg.content} for msg in history.messages]
    
    return ChatResponse(answer=response['answer'], chat_history=history_dicts)

@app.get("/")
def read_root():
    return {"Hello": "This is the RAG Backend"}