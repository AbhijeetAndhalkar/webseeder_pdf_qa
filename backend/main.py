import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import AskRequest, AskResponse
from rag import process_document, answer_question

# The traceback module is used to print the full error stack trace to the terminal, which is critical for debugging issues.
app = FastAPI(title="Document Q&A API") # Initializes the FastAPI application, serving as the core HTTP engine of our backend.

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking of uploaded document IDs to handle 404s properly.
# We use a set() here because looking up an ID in a set has O(1) time complexity (extremely fast).
valid_document_ids = set()

@app.get("/health")
async def health_check():
    """Returns the API status and configured models."""
    return {
        "status": "ok",
        "llm_model": "phi3",
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a .pdf file, temporarily saves it, processes it, and cleans up."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are allowed.")
        
    # Create a temporary file in the SYSTEM temp folder (outside your project)
    # This prevents Uvicorn and Live Server from detecting a change!
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_file_path = tmp.name

    try:
        doc_id, total_chunks = process_document(temp_file_path, file.filename)
        valid_document_ids.add(doc_id) # Adds the newly generated unique document ID to our tracker set
        
        # We hand the temp file to the rag.py logic, get a unique ID back, save that ID to our set, 
        # and tell the frontend it was successful by returning this JSON response.
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "total_chunks": total_chunks
        }
    except Exception as e:
        print(f"Error processing document: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error while processing the document.")
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# response_model=AskResponse guarantees the outgoing JSON data strictly matches the Pydantic schema defined in models.py.
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Answers a question based on the document context."""
    # 400: Question is empty
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
        
    # 404: Document not found
    if request.document_id not in valid_document_ids:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # 503: LLM call fails
    try:
        answer, sources = answer_question(request.document_id, request.question)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"LLM Call Failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="LLM call failed.")
# To run this FastAPI application locally, use the command: `uvicorn main:app --reload`
# The `/ask` endpoint sends the doc_id and question to the answer_question function in rag.py, which returns the answer and the source chunks used.