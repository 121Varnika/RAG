import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ollama import Client

# ------------- CONFIG -------------
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "bhel_report"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:1b"
# -----------------------------------

def ask(question, k=8):
    # load embedder
    embedder = SentenceTransformer(EMBED_MODEL)

    # load chroma client
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # embed the query
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32").tolist()

    # search top-k
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # build context string
    context_parts = []
    for doc, meta, dist in zip(docs, metas, dists):
        page = meta.get("page", "?")
        context_parts.append(f"[page {page}] (distance: {dist:.4f})\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # DEBUG — show what we retrieved
    print("------ Retrieved context (top-k) ------\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        page = meta.get("page", "?")
        print(f"Chunk {i} — page {page} — dist {dist:.4f}\n{doc[:500]}\n")
    print("------ end context ------\n")

    prompt = f"""
You are a helpful assistant. Use only the following extracted context to answer the question.
Do NOT add or assume anything that is not present in the context.

Context:
{context}

Question: {question}

Answer **only** from the context above.
""".strip()

    # call Ollama
    client = Client()
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    user_q = input("Your question: ")
    answer = ask(user_q)
    print("\n--- Answer ---\n")
    print(answer)