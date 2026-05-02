from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from backend.config import embedding_model, CHROMA_PATH
import os


def process_pdf(file_path: str) -> str:
    filename = os.path.basename(file_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # ── Split ─────────────────────────────────────────────────────────────────
    # Larger chunks (800) preserve more context per retrieval hit.
    # Overlap (150) ensures sentences at boundaries aren't lost.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # ── Tag metadata ──────────────────────────────────────────────────────────
    for chunk in chunks:
        chunk.metadata["source"] = filename

    # ── Upsert into Chroma ────────────────────────────────────────────────────
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    vectordb.add_documents(chunks)
    vectordb.persist()

    print(f"[ingestion] {filename} → {len(chunks)} chunks stored.")
    return "PDF processed successfully"