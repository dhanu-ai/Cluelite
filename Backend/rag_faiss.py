import os
import uuid
import logging
import pickle
import numpy as np
from typing import List, Optional, Tuple, Dict
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import aiofiles
from datetime import datetime

logger = logging.getLogger("Cluelite.RAG")

# Initialize components
embedder = SentenceTransformer('all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# FAISS index and document store
faiss_index = None
document_store = {}
index_path = "./faiss_index"
docstore_path = "./document_store.pkl"

def initialize_faiss():
    """Initialize or load FAISS index and document store."""
    global faiss_index, document_store
    
    # Create index directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    if os.path.exists(index_path) and os.path.exists(docstore_path):
        # Load existing index and document store
        try:
            faiss_index = faiss.read_index(index_path)
            with open(docstore_path, 'rb') as f:
                document_store = pickle.load(f)
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            create_new_index()
    else:
        create_new_index()

def create_new_index():
    """Create a new FAISS index."""
    global faiss_index, document_store
    dimension = 384  # all-MiniLM-L6-v2 embedding dimension
    faiss_index = faiss.IndexFlatL2(dimension)
    document_store = {}
    logger.info("Created new FAISS index")

def save_index():
    """Save FAISS index and document store to disk."""
    try:
        faiss.write_index(faiss_index, index_path)
        with open(docstore_path, 'wb') as f:
            pickle.dump(document_store, f)
        logger.info("Saved FAISS index and document store")
    except Exception as e:
        logger.error(f"Failed to save index: {e}")

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from PDF file asynchronously."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

def chunk_text(text: str) -> List[str]:
    """Split text into manageable chunks."""
    return text_splitter.split_text(text)

async def add_documents_to_db(file_path: str, filename: str) -> Tuple[str, int]:
    """Process PDF and add to FAISS database."""
    try:
        # Extract text
        text = await extract_text_from_pdf(file_path)
        
        # Split into chunks
        chunks = chunk_text(text)
        
        if not chunks:
            return "error: No text could be extracted from PDF", 0
        
        # Generate embeddings
        embeddings = embedder.encode(chunks)
        
        # Add to FAISS index and document store
        start_id = faiss_index.ntotal
        ids = list(range(start_id, start_id + len(chunks)))
        
        # Add to FAISS index
        faiss_index.add(embeddings)
        
        # Add to document store
        for i, (chunk_id, chunk) in enumerate(zip(ids, chunks)):
            document_store[chunk_id] = {
                "text": chunk,
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "embedding_id": i
            }
        
        # Save index
        save_index()
        
        logger.info(f"Added {len(chunks)} chunks from {filename} to database")
        return "success", len(chunks)
    except Exception as e:
        logger.error(f"Failed to add documents to DB: {e}")
        return f"error: {str(e)}", 0

def query_documents(query: str, n_results: int = 3) -> List[dict]:
    """Query the FAISS database for relevant document chunks."""
    try:
        if faiss_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedder.encode([query])
        
        # Search in FAISS index
        distances, indices = faiss_index.search(query_embedding, n_results)
        
        # Retrieve documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx in document_store and idx != -1:
                result = document_store[idx].copy()
                result["similarity_score"] = float(1 / (1 + distance))  # Convert distance to similarity
                results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Document query failed: {e}")
        return []

def clear_documents():
    """Clear all documents from the FAISS database."""
    try:
        global faiss_index, document_store
        create_new_index()
        save_index()
        logger.info("Cleared all documents from FAISS database")
        return "success"
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        return f"error: {str(e)}"

def get_document_stats():
    """Get statistics about the documents in the database."""
    return {
        "total_chunks": faiss_index.ntotal,
        "total_documents": len(set(doc["filename"] for doc in document_store.values())),
        "document_list": list(set(doc["filename"] for doc in document_store.values()))
    }

# Initialize FAISS on module import
initialize_faiss()