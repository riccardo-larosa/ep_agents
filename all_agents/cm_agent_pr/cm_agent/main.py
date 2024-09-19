import os, pprint

from langchain_core.tools import tool
#from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
#from typing import Union, Literal
from cm_agent.utils.nodes import retrieve_documents, grade_retrieved_documents, generate_response, rewrite_query, decide_to_rewrite_query
from cm_agent.utils.state import GraphState



TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"

# Define the graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve_documents)  # retrieve
workflow.add_node("grade_documents", grade_retrieved_documents)  # grade documents
workflow.add_node("generate", generate_response)  # generatae
workflow.add_node("rewrite_query", rewrite_query)  # transform_query

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_rewrite_query,
    {"rewrite_query": "rewrite_query", "generate": "generate"},
)
workflow.add_edge("rewrite_query", "retrieve")  # re-retrieve
workflow.set_finish_point("generate")

graph = workflow.compile()
