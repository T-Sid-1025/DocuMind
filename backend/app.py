from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil, os

from backend.ingestion import process_pdf
from backend.graph import build_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
os.makedirs("data", exist_ok=True)

uploaded_files: list[str] = []
chat_sessions:  dict[str, list] = {}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    process_pdf(file_path)
    if file.filename not in uploaded_files:
        uploaded_files.append(file.filename)
    return {"message": "PDF uploaded successfully", "files": uploaded_files}


@app.get("/files")
def get_files():
    return {"files": uploaded_files}


# ── Serve PDF for in-app preview ──────────────────────────────────────────────
@app.get("/pdf/{filename}")
def serve_pdf(filename: str):
    file_path = f"data/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="application/pdf")


@app.post("/ask")
async def ask_question(query: str, file: str = None):
    session_key = file if file else "general"
    history     = chat_sessions.get(session_key, [])

    state = {
        "query":        query,
        "file":         file or None,
        "chat_history": history
    }

    result = graph.invoke(state)

    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": result["answer"]})
    chat_sessions[session_key] = history

    return {
        "answer":     result["answer"],
        "confidence": result["confidence"],
        "decision":   result["decision"],
        "context":    result.get("context", ""),
        "sources":    result.get("sources", [])
    }


@app.get("/history")
def get_history(file: str = None):
    return {"history": chat_sessions.get(file or "general", [])}


@app.delete("/history")
def clear_history(file: str = None):
    chat_sessions[file or "general"] = []
    return {"message": "History cleared"}