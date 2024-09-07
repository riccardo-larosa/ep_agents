from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, List, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
import operator
from langchain_core.prompts import ChatPromptTemplate

class CM_search(BaseModel):
    """
    Use Commerce Manager search for questions that are related to Elastic Path and anything that does not require an API.
    """

    query: str = Field(description="The query to use when searching for Commerce Manager vectorstore.")


class API_search(BaseModel):
    """
    A search containing documents related to APIs for Elastic Path.
    """

    query: str = Field(description="The query to use when searching the API vectorstore.")

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Planner
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Router
# Preamble
preamble = """You are an expert at routing a user question to a search for Commerce Manager 
a search for APIs with OpenAPIs.
Commerce Manager search contains document related to using the platform without deep technical knowledge.
API search is for developers that want to understand what is the appropriate structure of APIs and how to call them. 
"""

# LLM with tool use and preamble
#llm = ChatOpenAI(model="command-r", temperature=0)
#structured_llm_router = llm.bind_tools(
#    tools=[CM_search, API_search], preamble=preamble
#)

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
    ]
)