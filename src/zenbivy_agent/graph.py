# src/zenbivy_agent/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
# importiere deine Tools/Helpers aus deiner bestehenden Datei (oder kopiere sie hierher)
# from .your_existing_module import SYSTEM, TOOLS, State, agent_node

# ---- (Minimal-Replikation aus deinem Code, aber OHNE InMemorySaver) ----
SYSTEM = "...dein SYSTEM Prompt..."  # oder von dir importieren
TOOLS = [wieder_verfuegbar, bedingungen, gear_guide, rag, search_web]  # importiert aus deiner Datei
llm = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=1, max_tokens=30000)
llm_with_tools = llm.bind_tools(TOOLS)

class State(MessagesState):
    pass

def agent_node(state: State, config: RunnableConfig):
    msgs = state["messages"]
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=SYSTEM)] + msgs
    ai = llm_with_tools.invoke(msgs, config=config)
    return {"messages": [ai]}

tool_node = ToolNode(TOOLS)
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# WICHTIG: ohne InMemorySaver kompilieren, damit der Server die Persistenz übernimmt
graph = builder.compile()
