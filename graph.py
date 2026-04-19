from __future__ import annotations
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
 
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from retriever import retrieve, list_available_files

# State schema
class RAGState(TypedDict):
    query: str
    reformed_query: str
    topic_filter: str | None # pdf name or none
    context: str
    chat_history: list[dict]
    ans: str

def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=2048,             
    )

def reformer(state: RAGState) -> RAGState:
    history = state.get("chat_history", [])
    query   = state["query"]
 
    if not history:
        return {**state, "reformed_query": query}
 
    llm = _get_llm()
 
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
    )   
    prompt = (
        "You are a document retrieval query optimizer for a multi-PDF academic assistant.\n"
        "Your task is to rewrite the LAST USER QUESTION into a structured standalone query "
        "optimized for retrieving the most relevant textbook content.\n\n"

        "The system has access to multiple PDFs (e.g., textbooks, notes). You MUST:\n"
        "1. Identify the CORE SUBJECT (e.g., Anatomy, Physiology, Pathology).\n"
        "2. Identify KEY TOPICS and SUBTOPICS.\n"
        "3. Infer the MOST RELEVANT DOCUMENT NAMES (based on typical textbook naming).\n"
        "4. Resolve pronouns using conversation history.\n"
        "5. Expand slightly for clarity, but DO NOT answer the question.\n\n"

        "OUTPUT FORMAT (STRICT):\n"
        "Query: <optimized search query>\n"
        "Subjects: <comma-separated subjects>\n"
        "Topics: <comma-separated key topics>\n"
        "Possible Documents: <comma-separated likely PDF names>\n\n"

        "Keep it concise and retrieval-focused.\n\n"

        f"Conversation History:\n{history_text}\n"
        f"Last User Question: {query}\n\n"
        "Output:"
    )

    result = llm.invoke([HumanMessage(content=prompt)])
    reformed = result.content.strip()
 
    return {**state, "reformed_query": reformed or query}

# Only for RETRIEVE
def retriever(state: RAGState) -> RAGState:
 
    query=state["reformed_query"],
    topic_filter=state.get("topic_filter")

    docs = retrieve(query=query, topic_filter=topic_filter)

    return {**state, "context": docs}

def generator(state: RAGState) -> RAGState:
    llm     = _get_llm()
    query   = state["reformed_query"]
    context = state.get("context", "")
    history = state.get("chat_history", [])
    files = list_available_files()
    files_txt = ", ".join(files) if files else "No PDFs loaded."

    if context:
        system_prompt = (
            "You are an academic tutor that answers questions using multiple textbooks (PDFs).\n"
            "You explain concepts clearly like a teacher, using only the provided context.\n\n"

            f"AVAILABLE FILES: {files_txt}\n"
            
            "These represent textbooks or notes you can reference.\n\n"

            f"DOCUMENT CONTEXT:\n{context}\n\n"

            "INSTRUCTIONS:\n"
            "1. First, internally determine which PDF(s) the context likely came from based on file names and topics.\n"
            "2. Use the context to TEACH the concept step-by-step, not just answer briefly.\n"
            "3. Structure answers like a textbook explanation:\n"
            "   - Definition\n"
            "   - Key Concepts\n"
            "   - Explanation\n"
            "   - Examples (if applicable)\n\n"

            "4. Always reference:\n"
            "   - File name (pdf_name)\n"
            "   - Topic/section headers if available\n\n"

            "5. DO NOT:\n"
            "   - Invent information\n"
            "   - Use knowledge outside the context\n\n"

            "6. If multiple documents are relevant, combine them logically.\n\n"

            "7. If the answer is NOT in the context:\n"
            "   Say: 'This topic is not covered in the available documents.'\n\n"

            "8. If the question is vague:\n"
            "   Ask a clarifying question instead of guessing.\n\n"

            f"User Question: {query}\n"
            "Teacher Response:"
        )
    else:
        system_prompt = (
            "You are a study assistant. Answer user queries with direct polite and concise answer."
        )
    
    messages = [SystemMessage(content=system_prompt)]

    for turn in history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
 
    messages.append(HumanMessage(content=query))
 
    result = llm.invoke(messages)

    print(f"[generator] context_len={len(context)}")
    
    return {**state, "ans": result.content.strip()}

def _build_graph() -> StateGraph:
    g = StateGraph(RAGState)
 
    g.add_node("reformer",  reformer)
    g.add_node("retriever",  retriever)
    g.add_node("generator",  generator)
    g.set_entry_point("reformer")
    g.add_edge("reformer", "retriever")
    g.add_edge("retriever", "generator")
    g.add_edge("generator", END)
 
    return g.compile()
 
 
_graph = None
 
def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph

# RAG Pipeline:
def run_rag(query: str, chat_history: list[dict]) -> dict:
    initial_state: RAGState = {
        "query": query,
        "reformed_query": "",
        "topic_filter": None,
        "context": "",
        "chat_history": chat_history,
        "ans": "",
    }

    final_state = get_graph().invoke(initial_state)
    return {
        "ans": final_state["ans"],
        "reformed_query": final_state["reformed_query"],
        "topic_filter": final_state.get("topic_filter"),
        "context": final_state.get("context", ""),
    }