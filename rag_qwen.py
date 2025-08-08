import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ------------------------
# CONFIG
# ------------------------
DATASET_PATH = "Dataset.json"  # your dataset file
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
TOP_K = 3  # number of records to retrieve

# ------------------------
# STEP 1: Load dataset
# ------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

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

# ------------------------
# STEP 2: Create embeddings & FAISS index
# ------------------------
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(docs, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} records.")

# ------------------------
# STEP 3: Retrieval function
# ------------------------
def retrieve_context(query, top_k=TOP_K):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    results = [docs[i] for i in indices[0]]
    return "\n\n".join(results)

# ------------------------
# STEP 4: Send to Qwen in LM Studio
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
        "model": "qwen",  # LM Studio will use the loaded Qwen3 14B model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(LM_STUDIO_API_URL, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ------------------------
# STEP 5: Test it
# ------------------------
if __name__ == "__main__":
    while True:
        q = input("\nEnter your medical question: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = ask_qwen(q)
        print("\nQwen's Answer:\n", answer)
