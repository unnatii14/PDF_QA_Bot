from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from groq import Groq
import os
import re
import uvicorn
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
import threading
from datetime import datetime

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# GROQ CLIENT SETUP
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it to your .env file.\n"
        "Get a free key at https://console.groq.com"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# GLOBAL STATE MANAGEMENT (Thread-safe)
# ---------------------------------------------------------------------------
# These variables track the current PDF session to prevent context leakage
vectorstore = None
qa_chain = False
current_pdf_session_id = None      # Unique ID for current PDF upload
current_pdf_upload_time = None     # Timestamp of when PDF was uploaded
pdf_state_lock = threading.RLock() # Thread-safe access to PDF state

# Load local embedding model (unchanged — FAISS retrieval stays the same)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION UTILITIES
# ---------------------------------------------------------------------------

def clear_vectorstore():
    """
    Safely clears the global vectorstore and resets PDF session state.
    This ensures complete isolation between PDF uploads.
    """
    global vectorstore, qa_chain, current_pdf_session_id, current_pdf_upload_time
    
    with pdf_state_lock:
        # Explicitly set to None to allow garbage collection
        vectorstore = None
        qa_chain = False
        current_pdf_session_id = None
        current_pdf_upload_time = None


def validate_pdf_session():
    """
    Validates that a PDF is currently loaded.
    Returns the current session ID if valid, None otherwise.
    """
    global current_pdf_session_id
    
    with pdf_state_lock:
        if not qa_chain or vectorstore is None or current_pdf_session_id is None:
            return None
        return current_pdf_session_id


def normalize_spaced_text(text: str) -> str:
    """
    Fixes character-level spaced text produced by PyPDFLoader on certain
    vector-based PDFs (e.g. NPTEL / IBM Coursera certificates).

    Examples:
        'J A I N I   S O L A N K I'  ->  'JAINI SOLANKI'
        'I B M'                       ->  'IBM'
        'N P T E L'                   ->  'NPTEL'

    Normal multi-letter words are left completely untouched.
    """
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")

    # Pattern: 3+ single alpha chars each separated by exactly one space
    pattern = r'\b(?:[A-Za-z] ){2,}[A-Za-z]\b'
    return re.sub(pattern, fix_spaced_word, text)


def normalize_answer(text: str) -> str:
    """
    Post-processes the LLM-generated answer:
    - Removes any residual character-level spacing.
    - Strips prompt leakage (lines starting with 'Answer', 'Context', etc.)
    - Collapses excessive whitespace.
    """
    # Remove residual character spacing in the answer itself
    text = normalize_spaced_text(text)
    # Strip any prompt-leakage prefixes the model might echo
    text = re.sub(r'^(Answer[^:]*:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
    # Collapse multiple spaces/newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# GROQ-BASED RESPONSE GENERATION
# ---------------------------------------------------------------------------

def generate_response(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Sends the prompt to the Groq API using the configured llama-3.3-70b-versatile
    model and returns the generated text.
    """
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=GROQ_MODEL,
        max_tokens=max_new_tokens,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------

class PDFPath(BaseModel):
    filePath: str

class AskRequest(BaseModel):
    question: str
    history: list = []


class SummarizeRequest(BaseModel):
    pdf: str | None = None


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: PDFPath):
    global vectorstore, qa_chain, current_pdf_session_id, current_pdf_upload_time
    
    try:
        with pdf_state_lock:
            # **CRITICAL**: Clear old vectorstore and session before processing new PDF
            # This is the primary fix for cross-document context leakage
            clear_vectorstore()
            
            # Create a new unique session ID for this PDF upload
            current_pdf_session_id = str(uuid.uuid4())
            current_pdf_upload_time = datetime.now().isoformat()
            
            loader = PyPDFLoader(data.filePath)
            raw_docs = loader.load()

            if not raw_docs:
                clear_vectorstore()
                return {"error": "PDF file is empty or unreadable. Please upload a valid PDF."}

            # ── Layer 1: normalize at ingestion ──────────────────────────────────────
            # Clean each page's text before chunking so embeddings are on real words.
            cleaned_docs = []
            for doc in raw_docs:
                cleaned_content = normalize_spaced_text(doc.page_content)
                cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(cleaned_docs)
            
            if not chunks:
                clear_vectorstore()
                return {"error": "No text chunks generated from the PDF. Please check your file."}

            # Create fresh vectorstore with only current PDF embeddings
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            qa_chain = True
            
            return {
                "message": "PDF processed successfully",
                "session_id": current_pdf_session_id,
                "upload_time": current_pdf_upload_time,
                "chunks_created": len(chunks)
            }
            
    except Exception as e:
        clear_vectorstore()
        return {
            "error": f"PDF processing failed: {str(e)}",
            "details": "Please ensure the file is a valid PDF"
        }


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    global vectorstore, qa_chain
    
    # **CRITICAL VALIDATION**: Ensure PDF is loaded and session is valid
    current_session = validate_pdf_session()
    if not current_session or not vectorstore or not qa_chain:
        return {"answer": "Please upload a PDF first!"}
    
    try:
        with pdf_state_lock:
            # Validate vectorstore still exists (safety check for concurrent requests)
            if vectorstore is None:
                return {"answer": "PDF session expired or cleared. Please upload a new PDF."}
            
            question = data.question
            history = data.history
            conversation_context = ""
            
            # Use only last 5 messages to avoid long prompts and context leakage
            # NEW: More defensive filtering of history
            if history:
                for msg in history[-5:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        conversation_context += f"{role}: {content}\n"
            
            docs = vectorstore.similarity_search(question, k=4)
            if not docs:
                return {"answer": "No relevant context found in the current PDF."}

            # ── Layer 2a: context is already clean (normalized at ingestion) ──────────
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""You are a helpful assistant answering questions ONLY from the provided PDF document.

Conversation History (for context only):
{conversation_context}

Document Context (ONLY reference this, NOT previous PDFs):
{context}

Current Question:
{question}

Instructions:
- Answer ONLY using the document context provided above.
- Do NOT use any information from previous documents or conversations outside this context.
- If the answer is not in the document, say so briefly.
- Keep the answer concise (2-3 sentences max).
- Do NOT mention previous PDFs or unrelated documents.

Answer:"""

            raw_answer = generate_response(prompt, max_new_tokens=512)

            # ── Layer 3: post-process the answer itself ────────────────────────────────
            answer = normalize_answer(raw_answer)
            return {"answer": answer}
            
    except Exception as e:
        return {"answer": f"Error processing question: {str(e)}"}


@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    global vectorstore, qa_chain
    
    # **CRITICAL VALIDATION**: Ensure PDF is loaded and session is valid
    current_session = validate_pdf_session()
    if not current_session or not vectorstore or not qa_chain:
        return {"summary": "Please upload a PDF first!"}

    try:
        with pdf_state_lock:
            # Validate vectorstore still exists (safety check for concurrent requests)
            if vectorstore is None:
                return {"summary": "PDF session expired or cleared. Please upload a new PDF."}

            docs = vectorstore.similarity_search("Give a concise summary of the document.", k=6)
            if not docs:
                return {"summary": "No document context available to summarize."}

            # Context is already clean (normalized at ingestion)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = (
                "You are a document summarization assistant working with a certificate or official document.\n"
                "RULES:\n"
                "1. Summarize in 6-8 concise bullet points.\n"
                "2. Clearly distinguish: who received the certificate, what course, which company issued it,\n"
                "   who signed it, on what platform, and on what date.\n"
                "3. Return clean, properly formatted text — no character spacing, proper Title Case for names.\n"
                "4. Use ONLY the information in the context below.\n"
                "5. DO NOT reference any other documents or previous PDFs.\n\n"
                f"Context:\n{context}\n\n"
                "Summary (bullet points):"
            )

            raw_summary = generate_response(prompt, max_new_tokens=512)
            summary = normalize_answer(raw_summary)
            return {"summary": summary}
            
    except Exception as e:
        return {"summary": f"Error summarizing PDF: {str(e)}"}


@app.post("/reset")
@limiter.limit("60/15 minutes")
def reset_session(request: Request):
    """
    Explicitly resets all PDF state and clears the vectorstore.
    Should be called by frontend when uploading a new PDF.
    """
    global vectorstore, qa_chain, current_pdf_session_id, current_pdf_upload_time
    
    with pdf_state_lock:
        old_session = current_pdf_session_id
        clear_vectorstore()
        return {
            "message": "Session cleared successfully",
            "cleared_session_id": old_session
        }


@app.get("/status")
def get_pdf_status(request: Request):
    """
    Returns the current PDF session status.
    Useful for debugging and ensuring proper state management.
    """
    global current_pdf_session_id, current_pdf_upload_time, qa_chain
    
    with pdf_state_lock:
        return {
            "pdf_loaded": qa_chain,
            "session_id": current_pdf_session_id,
            "upload_time": current_pdf_upload_time
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
