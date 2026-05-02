from langgraph.graph import StateGraph, END
from backend.rag_pipeline import get_answer

NO_ANSWER_PHRASES = [
    "i don't have enough information",
    "i don't know",
    "not mentioned",
    "not found in the document",
    "cannot find",
    "no information",
]


def process_node(state: dict) -> dict:
    result = get_answer(
        state["query"],
        state.get("file"),
        state.get("chat_history", [])   # ✅ pass memory
    )
    state["answer"]     = result["answer"]
    state["confidence"] = result["confidence"]
    state["context"]    = result.get("context", "")
    state["sources"]    = result.get("sources", [])
    return state


def grader_node(state: dict) -> dict:
    answer     = state.get("answer", "").lower()
    confidence = state.get("confidence", 1.0)
    no_info    = any(p in answer for p in NO_ANSWER_PHRASES)
    state["decision"] = "escalate" if (no_info or confidence < 0.25) else "answer"
    return state


def hitl_node(state: dict) -> dict:
    state["answer"] = (
        "I couldn't find a clear answer in the document for your question. "
        "A human support agent has been notified and will assist you shortly."
    )
    state["confidence"] = 0.0
    return state


def build_graph():
    graph = StateGraph(dict)
    graph.add_node("process", process_node)
    graph.add_node("grader",  grader_node)
    graph.add_node("hitl",    hitl_node)
    graph.set_entry_point("process")
    graph.add_edge("process", "grader")
    graph.add_conditional_edges(
        "grader",
        lambda state: state["decision"],
        {"answer": END, "escalate": "hitl"}
    )
    graph.add_edge("hitl", END)
    return graph.compile()