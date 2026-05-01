import uuid
from typing import Tuple, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
# A Document is just a custom LangChain container. It has two compartments: one for the text (page_content), and one for the sticky notes (metadata).
# Initialize local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize in-memory Chroma vector store
vector_store = Chroma(
    collection_name="document_qa_collection",
    embedding_function=embeddings
)

# Initialize local Ollama LLM (phi3)
llm = ChatOllama(model="phi3", temperature=0)

def process_document(file_path: str, filename: str) -> Tuple[str, int]:
    """
    Reads the PDF file using PyPDFLoader, splits the text into chunks, embeds them, 
    and stores them in ChromaDB.
    Returns a unique document ID and the total number of chunks.
    """
    doc_id = str(uuid.uuid4())
    
    # Load and parse the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Combine the text from all pages
    file_content = "\n".join([page.page_content for page in pages])
    
    # Split into manageable chunks (~500 chars with 50 overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    texts = text_splitter.split_text(file_content)
    
    if not texts:
        return doc_id, 0
        
    # We create a list of Document objects. Each Document contains a small chunk of text along with its metadata (sticky notes).
    docs = []
    for i, text in enumerate(texts):
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_number": i + 1
        }
        docs.append(Document(page_content=text, metadata=metadata))
    
    # Store vectors in ChromaDB
    if docs:
        vector_store.add_documents(docs)
    
    return doc_id, len(docs)

def answer_question(doc_id: str, question: str) -> Tuple[str, List[int]]:
    """
    Retrieves top chunks for the document and answers strictly using local Ollama.
    Returns the answer string and a list of source chunk numbers.
    """
    # Retrieve top 3 chunks matching the doc_id filter
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"doc_id": doc_id}
        }
    )
    
    retrieved_docs = retriever.invoke(question)
    
    if not retrieved_docs:
        return "The answer is not available in the document.", []
        
    context_texts = []
    source_chunks = []
    
    for doc in retrieved_docs:
        context_texts.append(doc.page_content)
        # Extract the chunk_number metadata from the Document object to show the user which chunks were used as sources
        if "chunk_number" in doc.metadata:
            source_chunks.append(doc.metadata["chunk_number"])
            
    # context_str is the combined text of the top 3 chunks retrieved from the ChromaDB vector store.
    context_str = "\n\n---\n\n".join(context_texts)
    
    # Strict prompt to prevent hallucinations
    prompt_template = """You are a helpful and strict assistant. Use the following context to answer the user's question. 
Rule: You must answer ONLY from the document context provided below. No general knowledge. 
If the answer is not in the context, say exactly: 'The answer is not available in the document.'

Context:
{context}

Question: {question}

Answer:"""

    prompt = prompt_template.format(context=context_str, question=question)
    
    # Send the strict prompt (which includes the 3 retrieved ChromaDB chunks as context) to the local phi3 model
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    
    # Clear sources if the model couldn't find the answer
    if answer == "The answer is not available in the document.":
        source_chunks = []
    else:
        source_chunks = sorted(list(set(source_chunks)))
    
    return answer, source_chunks
