from langchain_community.vectorstores import Chroma
from backend.config import embedding_model, CHROMA_PATH
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a precise Customer Support AI assistant.
Your ONLY job is to answer questions using the DOCUMENT CONTEXT provided below.

STRICT RULES:
1. Answer ONLY from the context. Never use outside knowledge.
2. If the exact answer is not in the context, respond EXACTLY with:
   "I don't have enough information in the document to answer this."
3. Be concise and clear — 2 to 5 sentences max.
4. Use bullet points only when listing multiple distinct items.
5. Never guess, assume, or make up any information.
6. If the user's question is vague, answer based on what is most relevant in the context.
7. Always sound like a helpful, professional support agent.
8. Do NOT start with "Based on the context" or "According to the document". Just answer directly.
9. You have access to the previous conversation. Use it to understand follow-up questions.
"""

NO_ANSWER_PHRASES = [
    "i don't have enough information",
    "i don't know",
    "not mentioned",
    "not found in the document",
    "cannot find",
    "no information",
]


def get_answer(query: str, selected_file: str = None, chat_history: list = None) -> dict:
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    search_kwargs = {"k": 6}
    if selected_file:
        search_kwargs["filter"] = {"source": selected_file}

    results = vectordb.similarity_search_with_score(query, **search_kwargs)

    if not results:
        return {
            "answer": "I don't have enough information in the document to answer this.",
            "confidence": 0.1,
            "context": "No matching content found.",
            "sources": []
        }

    SIMILARITY_THRESHOLD = 1.0
    strong = [(doc, score) for doc, score in results if score <= SIMILARITY_THRESHOLD]
    top_docs = strong[:5] if strong else results[:3]

    context_parts = []
    sources = []

    for doc, score in top_docs:
        context_parts.append(doc.page_content.strip())
        page = doc.metadata.get("page", None)
        source = doc.metadata.get("source", "Unknown")
        sources.append({
            "file": source,
            "page": int(page) + 1 if page is not None else None,
            "snippet": doc.page_content.strip()[:180]
        })

    context = "\n\n---\n\n".join(context_parts)

    # Multi-turn: build full message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for turn in chat_history[-6:]:   # last 6 turns = 3 Q&A pairs
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"DOCUMENT CONTEXT:\n{context}\n\nCUSTOMER QUESTION:\n{query}"
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        print("GROQ ERROR:", str(e))
        return {
            "answer": "Server error while generating answer. Please try again.",
            "confidence": 0.1,
            "context": str(e),
            "sources": []
        }

    has_answer = not any(p in answer.lower() for p in NO_ANSWER_PHRASES)
    confidence = round(0.9 - (top_docs[0][1] / 10), 2) if has_answer else 0.2
    confidence = max(0.1, min(confidence, 0.99))

    return {
        "answer": answer,
        "confidence": confidence,
        "context": context[:400],
        "sources": sources
    }