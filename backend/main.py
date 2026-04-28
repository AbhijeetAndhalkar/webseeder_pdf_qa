import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import AskRequest, AskResponse
from rag import process_document, answer_question

app = FastAPI(title="Document Q&A API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking of uploaded document IDs to handle 404s properly
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
    # 415: Wrong file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Wrong file type. Only .pdf files are allowed.")
        
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Temporarily save the uploaded PDF to disk
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
            
        # Pass the file path to our updated rag.py (PyPDFLoader)
        doc_id, total_chunks = process_document(temp_file_path, file.filename)
        valid_document_ids.add(doc_id)
        
        # Instantly delete the temporary file after successful parsing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "total_chunks": total_chunks
        }
    except Exception as e:
        # Ensure we delete the file even if an error occurs
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        print(f"Error processing document: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error while processing the document.")

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
