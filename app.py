import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import torch
import asyncio
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

# Load models with caching
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16  # Optimize for GPU
        )
        return embedder, tokenizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Process document with error handling
def process_document(file) -> List[str]:
    try:
        if file.name.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            text = file.read().decode('utf-8')
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_text(text)
        if not chunks:
            raise ValueError("No text extracted from document")
        return chunks
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    try:
        client = QdrantClient(":memory:")  # In-memory for simplicity
        client.create_collection(
            collection_name="doc_chunks",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Qdrant: {e}")
        return None

# Embed and store chunks with batch processing
def embed_and_store(chunks: List[str], embedder, client):
    embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)
    client.upsert(
        collection_name="doc_chunks",
        points=[
            models.PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk})
            for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
    )
    return embeddings  # Return for BM25 setup

# Hybrid retrieval (dense + sparse)
async def retrieve_chunks(query: str, embedder, client, chunks: List[str], embeddings: np.ndarray, k=3) -> List[str]:
    # Dense retrieval with Qdrant
    query_embedding = embedder.encode([query])[0]
    dense_results = client.search(
        collection_name="doc_chunks",
        query_vector=query_embedding.tolist(),
        limit=k * 2  # Get more for re-ranking
    )
    dense_chunks = [point.payload["text"] for point in dense_results]
    dense_scores = [point.score for point in dense_results]

    # Sparse retrieval with BM25
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]
    sparse_chunks = [chunks[i] for i in top_bm25_indices]
    sparse_scores = [bm25_scores[i] for i in top_bm25_indices]

    # Combine and re-rank
    combined_chunks = list(set(dense_chunks + sparse_chunks))
    combined_scores = {}
    for chunk in combined_chunks:
        d_score = dense_scores[dense_chunks.index(chunk)] if chunk in dense_chunks else 0
        s_score = sparse_scores[sparse_chunks.index(chunk)] if chunk in sparse_chunks else 0
        combined_scores[chunk] = 0.7 * d_score + 0.3 * s_score  # Weighted combination
    
    sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:k]]

# Generate response with structured output
def generate_response(query: str, context_chunks: List[str], tokenizer, model) -> str:
    prompt = (
        "You are a helpful assistant. Use only the following document excerpts to answer the question. "
        "Do not invent information or use external knowledge. If the answer is unclear, say so. "
        "Format the response as a concise paragraph or bullet points based on the query. "
        f"Document excerpts: {' '.join(context_chunks)} "
        f"Question: {query} "
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    
    # Post-process for better formatting
    if "list" in query.lower() or "points" in query.lower():
        lines = response.split(". ")
        response = "\n".join([f"- {line.strip()}" for line in lines if line.strip()])
    return response

# Streamlit UI
st.title("Enhanced RAG Document Chatbot")

uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        chunks = process_document(uploaded_file)
        if not chunks:
            st.stop()
        
        embedder, tokenizer, model = load_models()
        if embedder is None:
            st.stop()
        
        client = init_qdrant()
        if client is None:
            st.stop()
        
        embeddings = embed_and_store(chunks, embedder, client)
        st.session_state['chunks'] = chunks
        st.session_state['embeddings'] = embeddings
    st.success("Document processed successfully!")
    
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating response..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            context_chunks = loop.run_until_complete(
                retrieve_chunks(query, embedder, client, st.session_state['chunks'], st.session_state['embeddings'])
            )
            response = generate_response(query, context_chunks, tokenizer, model)
        st.markdown(response)
