import os
import json
import faiss
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ------------------------
# CONFIG
# ------------------------
DATASET_PATH = "Data.json"         # Your dataset file
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
TOP_K = 3                                  # Number of records to retrieve
FAISS_INDEX_PATH = "my_faiss_index.index"  # Path to save/load the FAISS index
DOCS_PATH = "my_docs.pkl"                  # Path to save/load the document list

# ------------------------
# STEP 1: Load dataset from JSON
# ------------------------
# This part is only needed if we are building the index for the first time.
# We will move the main logic into the 'if/else' block in Step 2.
print("STEP 1: Data loading configuration complete.")

# ------------------------
# STEP 2: Create or Load embeddings & FAISS index
# ------------------------
# This is the core of our update. We check if the saved files exist.

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCS_PATH):
    # --- This block runs if we have already created and saved the index ---
    print("\nSTEP 2: Loading existing FAISS index and documents... (The Fast Lane)")
    
    # Load the FAISS index from disk
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load the document list from disk
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
        
    # We still need to load the embedding model to encode user queries
    embedder = SentenceTransformer(EMBED_MODEL)
    
    print("Loading complete.")

else:
    # --- This block runs ONLY the first time, or if you delete the saved files ---
    print("\nSTEP 2: Building new FAISS index from scratch... (The First Time Setup)")

    # Load the full dataset from JSON
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Process the text from the dataset
    docs = []
    for entry in dataset:
        combined_text = f"""
Condition: {entry['Condition']}
Symptoms: {entry['Symptoms']}
Drug Name: {entry['Drug_Name']}
Dosage: {entry['Dosage']}
Side Effects: {entry['Side_Effects']}
Warning: {entry['Warning']}
"""
        docs.append(combined_text)
    print(f"Loaded {len(docs)} medical records.")

    # Load the embedding model to create vectors
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Creating embeddings... (This may take a moment)")
    embeddings = embedder.encode(docs, convert_to_numpy=True)

    # Build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # --- SAVE THE INDEX AND DOCS FOR FUTURE USE ---
    print("Saving FAISS index and documents for future runs...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
        
    print(f"FAISS index built and saved with {index.ntotal} records.")


# ------------------------
# STEP 3: Retrieval function (No changes needed)
# ------------------------
def retrieve_context(query, top_k=TOP_K):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    results = [docs[i] for i in indices[0]]
    return "\n\n".join(results)

# ------------------------
# STEP 4: Send to Qwen in LM Studio (No changes needed)
# ------------------------
def ask_qwen(user_query):
    context = retrieve_context(user_query)
    prompt = f"""You are a medical assistant.
Use ONLY the following medical records to answer the question.
If you are unsure, say you don't know.

Context:
{context}

User Question: {user_query}
"""

    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    # Add error handling for the request
    try:
        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to LM Studio at {LM_STUDIO_API_URL}. Please ensure it is running and the server is on."


# ------------------------
# STEP 5: Test it (No changes needed)
# ------------------------
if __name__ == "__main__":
    print("\nSTEP 5: Starting interactive session.")
    while True:
        q = input("\nEnter your medical question: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = ask_qwen(q)
        print("\nQwen's Answer:\n", answer)