import os
import json
import fitz #PyMuPDF
import faiss
import numpy as np
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
import pdfplumber
import tiktoken





# ------------------------------------------------
# CONFIG
# ------------------------------------------------
OPENAI_API_KEY = ""
DATA_FOLDER = "/backup/Programs/Sumit_data/tmp1"
TOP_K_DENSE = 1
TOP_K_SPARSE = 1
TOP_K_FINAL = 1

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------
# READ MULTIMODAL FILES
# ------------------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

def read_multimodal_folder(folder_path):
    docs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = file.lower().split(".")[-1]

            try:
                if ext == "pdf":
                    text = extract_text_from_pdf(path)
                    docs.append({
                        "id": len(docs),
                        "path": path,
                        "type": "pdf",
                        "text": text
                    })

                elif ext in ["txt", "md"]:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append({
                        "id": len(docs),
                        "path": path,
                        "type": "text",
                        "text": text
                    })

                elif ext in ["jpg", "jpeg", "png"]:
                    docs.append({
                        "id": len(docs),
                        "path": path,
                        "type": "image",
                        "text": f"Image file: {os.path.basename(path)}"
                    })

            except Exception as e:
                print(f"Error reading {path}: {e}")

    return docs

documents = read_multimodal_folder(DATA_FOLDER)
print(f"Loaded {len(documents)} documents")

# ------------------------------------------------
# DENSE EMBEDDINGS (text-embedding-3-large)
# ------------------------------------------------
def chunk_text_by_tokens(text, model_name="text-embedding-3-large", 
                         max_tokens=1000, overlap=200):
    """Splits text into chunks based on token count."""
    # text-embedding-3-large uses the cl100k_base tokenizer profile
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        # Move forward by max_tokens minus overlap to keep semantic continuity
        start += (max_tokens - overlap) if (end < len(tokens)) else max_tokens
        
    return chunks

def get_embedding(path):
    text = extract_text_from_pdf(path) if path.lower().endswith(".pdf") else ""

    print(text)
    text = text[:1000]  # safety limit
    
    text_chunks = chunk_text_by_tokens(text, model_name="text-embedding-3-large",
                                        max_tokens=1000, overlap=200)
    
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text_chunks
    )
    embedding = [item.embedding for item in response.data]
    return embedding

dense_vectors = []

for doc in documents:
    emb = get_embedding(doc['path'])
    for e in emb:
        dense_vectors.append(e)

dense_vectors = np.array(dense_vectors, dtype=np.float32)

# ------------------------------------------------
# FAISS DENSE INDEX
# ------------------------------------------------
dimension = dense_vectors.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)

# Normalize for cosine similarity
faiss.normalize_L2(dense_vectors)
faiss_index.add(dense_vectors)
faiss.write_index(faiss_index, "my_faiss_store")
print("Dense index created")

# ------------------------------------------------
# SPARSE BM25 INDEX
# ------------------------------------------------
tokenized_docs = [get_embedding(doc["path"]).lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
BM25Okapi.save(bm25, "my_bm25_store")
print("Sparse BM25 index created")

# ------------------------------------------------
# RE-RANKER
# ------------------------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------
# HYBRID RETRIEVAL
# ------------------------------------------------
def hybrid_retrieve(query):
    faiss_index = faiss.read_index("my_faiss_store")
    bm25 = BM25Okapi.load("my_bm25_store")

    # -------- Dense retrieval --------
    q_emb = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(q_emb)

    dense_scores, dense_ids = faiss_index.search(q_emb, TOP_K_DENSE)

    # -------- Sparse retrieval --------
    sparse_scores = bm25.get_scores(query.lower().split())

    # Normalize sparse scores
    sparse_scores = np.array(sparse_scores)
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()

    # -------- Score fusion --------
    final_scores = {}

    for rank, doc_id in enumerate(dense_ids[0]):
        final_scores[doc_id] = final_scores.get(doc_id, 0) + dense_scores[0][rank]

    for doc_id, score in enumerate(sparse_scores):
        final_scores[doc_id] = final_scores.get(doc_id, 0) + score

    # Top candidates
    candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    candidate_docs = [documents[idx] for idx, _ in candidates]

    # -------- Re-ranking --------
    pairs = [(query, doc["text"][:1000]) for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K_FINAL]

    return reranked

# ------------------------------------------------
# CREATE MULTIMODAL PROMPT
# ------------------------------------------------
def build_multimodal_prompt(query, retrieved_docs):
    context_parts = []
    image_paths = []

    for doc, score in retrieved_docs:
        if doc["type"] == "image":
            image_paths.append(doc["path"])
        else:
            context_parts.append(
                f"[Source: {doc['path']}]\n{doc['text'][:1000]}"
            )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a legal-domain AI assistant.

Answer the user's question using ONLY the provided context.
If the answer is not present, say so explicitly.

USER QUESTION:
{query}

TEXT CONTEXT:
{context}

IMAGE FILES AVAILABLE:
{image_paths}

Provide:
1. Direct answer
2. Key legal findings
3. Source references
4. Any ambiguity or missing information
"""

    return prompt, image_paths

# ------------------------------------------------
# EXAMPLE QUERY
# ------------------------------------------------
query = "Who is Ayan?"

results = hybrid_retrieve(query)

print("\nTop Retrieved Documents:")
for doc, score in results:
    print(f"Score={score:.4f} | {doc['path']}")

prompt, images = build_multimodal_prompt(query, results)

print("\nFINAL PROMPT:\n")
print(prompt[:1000])