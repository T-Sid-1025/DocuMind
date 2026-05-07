from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from backend.config import embedding_model, CHROMA_PATH
import os


def process_file(file_path: str) -> str:
    filename  = os.path.basename(file_path)
    extension = filename.split(".")[-1].lower()

    # ── Pick the right loader ─────────────────────────────────────────────────
    if extension == "pdf":
        loader = PyPDFLoader(file_path)

    elif extension in ["docx", "doc"]:
        loader = Docx2txtLoader(file_path)

    elif extension in ["xlsx", "xls"]:
        loader = UnstructuredExcelLoader(file_path, mode="elements")

    else:
        print(f"[ingestion] Unsupported file type: {extension}")
        return f"Unsupported file type: {extension}"

    # ── Load ──────────────────────────────────────────────────────────────────
    documents = loader.load()

    # ── Split ─────────────────────────────────────────────────────────────────
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

    print(f"[ingestion] {filename} ({extension}) → {len(chunks)} chunks stored.")
    return "File processed successfully"