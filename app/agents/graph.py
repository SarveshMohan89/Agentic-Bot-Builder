import json
from typing import Any, Dict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.models.schemas import AgentState
from app.core.vector_store import get_vector_store
from app.core.config import get_settings

settings = get_settings()

def __getllm() -> ChatAnthropic:
    return ChatAnthropic(
        model = settings.claude_model,
        anthropic_api_key = settings.anthropic_api_key,
        temperature = 0.3,
        max_tokens = 1024,
    )

def query_router_agent(state: AgentState) -> AgentState:
    state.agent_trace.append("QueryRouterAgent")
    llm = __getllm()

    system = """You are a query analysis expert. Analyse the user question and respond with JSON only.
    
Respond ONLY with this JSON and nothing else:
    {
        "intent" : "<one of: factual|how-to|comparison|definition|troubleshooting|other>",
        "rewritten_query" : "<improved query for semantic search>",
        "key_concepts" : ["<concept1>", "<concept2>"]
    }"""

    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"User Question: {state.question}")
        ])
        data = json.loads(response.content.strip())
        state.query_intent = data.get("intent", "other")
        rewritten = data.get("rewritten_query", state.question)
        if rewritten and len(rewritten) > 3:
            state.question = rewritten

    except Exception as e:
        state.agent_trace.append(f"QueryRouterAgent Warning: {e}")
        state.query_intent = "other"

    return state

def rag_retrieval_agent(state : AgentState) -> AgentState:
    state.agent_trace.append("RAGRetrievalAgent")
    vector_store = get_vector_store()

    try:
        docs = vector_store.similarity_search(
            bot_id = state.bot_id,
            query = state.question,
            top_k = settings.top_k_results,
            score_threshold = 0.2,
        )
        state.retrieved_docs = docs

        if docs:
            top_scores = [d["score"] for d in docs[:2]]
            state.retrieval_confidence = round(
                sum(top_scores) / len(top_scores), 4
            )

        else:
            state.retrieval_confidence = 0.0

        state.agent_trace.append(
            f"RAGRetrievalAgent : found {len(docs)} docs, "
            f"confidence = {state.retrieval_confidence}"
        )

    except Exception as e:
        state.agent_trace.append(f"RAGRetrievalAgent ERROR: {e}")
        state.retrieved_docs = []
        state.retrieval_confidence = 0.0

    return state

def route_by_confidence(state : AgentState) -> Literal["answer", "fallback"]:
    threshold = settings.similarity_threshold
    strong_docs = [
        d for d in state.retrieved_docs
        if d["score"] >= threshold
    ]
    if len(strong_docs) >= 1 and state.retrieval_confidence >= threshold:
        return "answer"

    return "fallback"

def answer_generator_agent(state: AgentState) -> AgentState:
    state.agent_trace.append("AnswerGeneratorAgent")
    llm = __getllm()

    context_parts = []
    for i, doc in enumerate(state.retrieved_docs[:3], 1):
        meta = doc.get("metadata", {})
        source_name = meta.get("source_name", "Unknown")
        context_parts.append(f"[Source{i}: {source_name}]\n{doc['text']}")

    context = "\n\n-----\n\n".join(context_parts)

    bot_config = state.bot_config or {}
    custom_system = bot_config.get("system_prompt", "")
    bot_domain = bot_config.get("domain", "the topic")

    base_system = f"""You are an intelligent assistant specialising in {bot_domain}.
    Answer questions ONLY based on the provided context.
    Be accurate, concise and helpful.
    Do NOT make up information not present in the context.
    {custom_system}
"""
    user_prompt = f"""CONTEXT FROM KNOWLEDGE BASE:
    {context}
    
    ---
    
    Conversation History:
    {_format_history(state.chat_history)}
    
    USER QUESTION: {state.question}
    
    Please provide a helpful and clear answer based on the context above.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=base_system),
            HumanMessage(content=user_prompt),
        ])
        state.answer = response.content.strip()
        state.answer_type = "direct"
        state.confidence = state.retrieval_confidence

    except Exception as e:
        state.agent_trace.append(f"AnswerGeneratorAgent ERROR: {e}")
        state.answer = "I encountered an error. Please try again."
        state.answer_type = "fallback"
        state.confidence = 0.0

    return state

def fallback_agent(state : AgentState) -> AgentState:
    state.agent_trace.append("FallbackAgent")
    vector_store = get_vector_store()

    try:
        related = vector_store.similarity_search(
            bot_id = state.bot_id,
            query = state.question,
            top_k = 3,
            score_threshold = 0.15,
        )
        state.related_docs = related

    except Exception:
        state.related_docs = []

    llm = __getllm()
    bot_config = state.bot_config or {}
    fallback_msg = bot_config.get(
        "fallback_message",
        "I could not find the appropriate answer, but here are some related topics."
    )

    if state.related_docs:
        topics = []
        for doc in state.retrieved_docs[:3]:
            meta = doc.get("metadata", {})
            name = meta.get("source_name", "Related Article")
            topics.append(f"-{name}")

        topics_str = "\n".join(topics)

        system = """You are a helpful assistant. You don't have a direct answer but want to guide the user to a related content. Be empathetic and helpful."""

        prompt = f"""The user asked: "{state.question}"
    I found these related topics:
{topics_str}

Write a brief response:
1. Acknowledge you don't have the exact answer.
2. Mentions the related topics that might help.
Keep it under 200 words.
"""
        try:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            state.answer = response.content.strip()

        except Exception:
            state.answer = fallback_msg

    else:
        state.answer = fallback_msg

    state.answer_type = "fallback" if not state.related_docs else "partial"
    state.confidence = max(state.retrieval_confidence, 0.1)
    return state

def citation_agent(state: AgentState) -> AgentState:
    state.agent_trace.append("CitationAgent")
    seen_urls = set()
    seen_names = set()
    sources = []

    for doc in state.retrieved_docs:
        meta = doc.get("meatdata", {})
        source_url = meta.get("source_url", "") or None
        source_name = meta.get("source_name", "Knowledge Base")
        dedup_key = source_url or source_name
        if dedup_key in seen_urls:
            continue

        seen_urls.add(dedup_key)
        seen_names.add(source_name)
        sources.append({
            "title" : source_name,
            "url" : source_url,
            "snippet" : doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"],
            "relevance_score" : doc["score"],
        })
    state.sources = sources[:3]

    related_topics = []
    related_pool = state.related_docs or state.related_docs[3:]

    for doc in related_pool:
        meta = doc.get("metadata", {})
        source_url = meta.get("source_url", "") or None
        source_name = meta.get("source_name", "Related Article")
        dedup_key = source_url or source_name
        if dedup_key in seen_urls or source_name in seen_names:
            continue

        seen_urls.add(dedup_key)
        related_topics.append({
            "title" : source_name,
            "url" : source_url,
            "snippet" : doc["text"][:75] + "..." if len(doc["text"]) > 75 else doc["text"],
            "relevance_Score" : doc["score"],
        })

    state.related_topics = related_topics[:2]
    return state

def _format_history(history : List[Dict[str, str]]) -> str:
    if not history:
        return "(No previous conversations found.)"

    lines = []
    for msg in history[-4:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("query_router", query_router_agent)
    graph.add_node("rag_retrieval", rag_retrieval_agent)
    graph.add_node("answer_generator", answer_generator_agent)
    graph.add_node("fallback", fallback_agent)
    graph.add_node("citation", citation_agent)

    graph.set_entry_point("query_router")
    graph.add_edge("query_router", "rag_retrieval")

    graph.add_conditional_edges(
        "rag_retrieval",
        route_by_confidence,
        {
            "answer" : "answer_generator",
            "fallback" : "fallback",
        },
    )

    graph.add_edge("answer_generator", "citation")
    graph.add_edge("fallback", "citation")
    graph.add_edge("citation", END)

    return graph.compile()

_compiled_graph = None

def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()

    return _compiled_graph

