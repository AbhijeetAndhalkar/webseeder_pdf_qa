# 100% Local Document Q&A AI

A completely offline, 100% local Document Q&A application. This project allows you to drag-and-drop a PDF file and ask questions about its contents, with all data processing and AI inference happening locally on your machine. No data is ever sent to the cloud, and no API keys are required.

## Architecture

*   **Backend:** FastAPI
*   **Vector Database:** ChromaDB (In-Memory)
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) - runs locally
*   **PDF Parsing:** LangChain `PyPDFLoader`
*   **Local LLM Inference:** Ollama (`phi3`) - runs locally on your GPU/CPU

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed. Navigate to the `backend` folder and install the required packages:

```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install and Start Ollama
You need [Ollama](https://ollama.com/) installed on your system.
Once installed, pull the `phi3` model (this will download the model to your machine):
```bash
ollama run phi3
```
*You can exit the Ollama prompt once it finishes downloading. The Ollama background service must remain running.*

### 3. Run the Backend Server
Start the FastAPI server from the `backend` directory:
```bash
uvicorn main:app --reload  
```
The API will be available at `http://127.0.0.1:8000`.

### 4. Open the Frontend
Simply double-click the `index.html` file inside the `frontend` folder to open it in your web browser. Drag and drop a `.pdf` file, wait for it to process, and start asking questions!
