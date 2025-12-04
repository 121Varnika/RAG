import os
import PyPDF2
import chromadb
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# ---------------- CONFIG ----------------
PDF_PATH = "C:\\Users\\121va\\OneDrive\\Desktop\\Internship BHEL\\data\\bhel annual report.pdf"        # change this to your PDF path
PERSIST_DIR = "./chroma_store"       # folder where DB is saved
COLLECTION_NAME = "bhel_report"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000            # characters per chunk
CHUNK_STEP = 800             # chunk_size - overlap (overlap = 200 chars)
BATCH = 64                   # embedding batch size
# -----------------------------------------


def chunk_text(page_text, page_no, chunk_size=CHUNK_SIZE, step=CHUNK_STEP):
    """Chunk text with overlap."""
    chunks, metas = [], []

    if not page_text:
        return chunks, metas

    for i in range(0, max(1, len(page_text)), step):
        chunk = page_text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
            metas.append({"page": page_no, "start": i})

    return chunks, metas


def pdf_to_chroma():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Reading PDF: {PDF_PATH}")
    with open(PDF_PATH, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        pages = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            pages.append((i + 1, text))

    print("Chunking text...")
    all_chunks = []
    all_metadata = []

    for page_no, txt in pages:
        c, m = chunk_text(txt, page_no)
        all_chunks.extend(c)
        all_metadata.extend(m)

    print(f"Total chunks created: {len(all_chunks)}")

    # Initialize Chroma persistent DB
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"source": os.path.basename(PDF_PATH)}
    )

    print("Embedding & storing chunks in ChromaDB...")

    # Process in batches
    for i in tqdm(range(0, len(all_chunks), BATCH)):
        batch_texts = all_chunks[i:i+BATCH]
        batch_metas = all_metadata[i:i+BATCH]

        emb = embedder.encode(batch_texts, convert_to_numpy=True).astype("float32")
        emb = emb.tolist()  # convert numpy → list for Chroma

        batch_ids = [f"chunk_{i+j}" for j in range(len(batch_texts))]

        collection.add(
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=emb,
            ids=batch_ids
        )

    print("\n✔ DONE! Vector database saved in:", PERSIST_DIR)


if __name__ == "__main__":
    pdf_to_chroma()