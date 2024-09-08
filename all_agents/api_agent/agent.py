import os, pprint


from typing import Annotated, List, Tuple, TypedDict
import operator
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing import Union, Literal
from langchain import hub

#from dotenv import load_dotenv
#load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"

###########################################
### Classes and Utils

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[List], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
def get_tools():
    tools = [TavilySearchResults(max_results=3)] 
    #tools = [ get_CM_answer] 
    return tools

def get_planner_agent():
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                SystemMessage(
                    content="""For the given objective, come up with a simple step by step plan. \
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
                    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""
                )
            ),
            ("placeholder", "{messages}"),
        ]
    )
    planner = planner_prompt | ChatOpenAI(
                model="gpt-4o", temperature=0
            ).with_structured_output(Plan)
    return planner
    
def get_execution_agent():
    tools = get_tools()
    #exec_prompt = ChatPromptTemplate.from_messages(
    #    [
    #        (
    #            SystemMessage(
    #                content="""You are a helpful assistant"""
    #            )
    #        ),
    #        ("placeholder", "{messages}"),
    #    ]
    #)
    exec_prompt = hub.pull("hwchase17/react")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent_executor = create_react_agent(llm, tools, messages_modifier=exec_prompt)
    return agent_executor

def get_replanner_agent():
    replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
                    Do not add any superfluous steps. \
                    The result of the final step should be the final answer. 
                    Make sure that each step has all the information needed - do not skip steps.

                    Your objective was this:
                    {input}

                    Your original plan was this:
                    {plan}

                    You have currently done the follow steps:
                    {past_steps}

                    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. 
                    Otherwise, fill out the plan. 
                    Only add steps to the plan that still NEED to be done. 
                    Do not return previously done steps as part of the plan."""
    )
    replanner = replanner_prompt | ChatOpenAI(
                model="gpt-4o", temperature=0
            ).with_structured_output(Action)
    return replanner

class Response(BaseModel):
    """Response to user."""

    response: str

class Action(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings

def get_CM_doc(prompt):
    """get retriever to query Elastic Path documentation for Commerce Manager"""
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db_name = "rag_db"
    collection_name = "epdocs_openaiembeddings"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"
    # Create a MongoDBAtlasVectorSearch object
    db = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        embeddings,
        index_name = vector_search_index
    )
    # TODO: I could 'pre-filter' the query if I know I want something specific from the docs 
    results = db.similarity_search_with_score(prompt, k=4)
    return results

############################################
### Tools
@tool
def get_CM_answer(question):
    """Use this to find the right information from Commerce Manager for Elastic Path queries that don't require an API.
        This tool is geared towards users who are looking for information on how to use Commerce Manager.
    """
    PROMPT_TEMPLATE = """
        \n\n\033[33m--------------------------\033[0m\n\n
        You are knowledgeable about Elastic Path products. You can answer any questions about 
        Commerce Manager, 
        Product Experience Manager also known as PXM,
        Cart and Checkout,
        Promotions,
        Composer,
        Payments
        Subscriptions,
        Studio.
        {prompt_base}
        Answer the question based only on the following context:
        \n\033[33m--------------------------\033[0m\n
        {context}
        \n\033[33m--------------------------\033[0m\n
        Answer the question based on the above context: {question}
        \n\033[33m--------------------------\033[0m\n
        """
    PROMPT_BASE = """
        Build any of the relative links using https://elasticpath.dev as the root
        """
    results = get_CM_doc(question)
    context_text = "\n\n\033[32m--------------------------\033[0m\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_context = prompt_template.format(prompt_base=PROMPT_BASE, context=context_text, question=question)
    model = ChatOpenAI(temperature=0.7, model="gpt-4o")
    response = model.invoke(prompt_context)

    return response.content 
############################################
### Nodes
async def create_plan(state: PlanExecute):
    """Come up with a simple step by step plan given the initial message
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        plan (dict): A list of steps to follow 
    """
    planner_agent = get_planner_agent()
    plan = await planner_agent.ainvoke({"messages": [("user", state["input"])]})
    #print(f"Here is the plan: {plan}")
    return {"plan": plan.steps}

async def execute_step(state: PlanExecute):
    """Execute the plan
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        past_steps (dict): Update all the steps that were executed
    """
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
            {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    # I need to fix this, it's giving me error: "Expected mapping type as input to PromptTemplate. Received <class 'list'>.""
    # agent_executor = get_execution_agent()
    print(task_formatted)
    #tools = [TavilySearchResults(max_results=3)]
    tools = get_tools()
    prompt = hub.pull("wfh/react-agent-executor")
    prompt.pretty_print()
    llm = ChatOpenAI(model="gpt-4o")
    agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    #print(response)
    agent_content = agent_response["messages"][-1].content
    print(f"Agent response: {agent_content}")
    #if not isinstance(agent_content, list):
    #    agent_content = [agent_content]
    return {
        #"past_steps": [task , agent_content],
        "past_steps": [task, agent_response["messages"][-1].content],
    }

async def replan_step(state: PlanExecute):
    """Replan the steps and see if we can up with a better plan
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        response (str): If no more steps are needed and you can return to the user, then respond with that. 
        plan (list): a new set of steps to follow  
    """
    replanner_agent = get_replanner_agent()
    print(f"State: {state}")
    output = await replanner_agent.ainvoke(state)
    print(f"$$$$$ Replanner output: {output}")
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}
############################################
### Conditions/edges
def should_end(state: PlanExecute) -> Literal["executor", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "executor"

############################################
### Graph
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(PlanExecute)
## nodes
workflow.add_node("planner", create_plan)
workflow.add_node("executor", execute_step)
workflow.add_node("replanner", replan_step)

## edges: start with planner
workflow.set_entry_point("planner")
# from plan we go to execute it
workflow.add_edge("planner", "executor")
# from executor we replan
workflow.add_edge("executor", "replanner")
# from replan we determine if we should end or go back to agent
workflow.add_conditional_edges("replanner", should_end,)

graph = workflow.compile()

"""
config = {"recursion_limit": 50}
input = {"input": "what is the hometown of the 2024 Australia open winner"}
import asyncio
async def run_workflow():
    async for output in graph.astream(input, config=config):
        for key, value in output.items():
            # Node
            #pprint.pprint(f"Output from node '{key}':")
            pprint.pprint(f"{key} ---")
            pprint.pprint(value)
            # Optional: print full state at each node
            if key != "__end__":
                print(value)
        print("\n---\n")

try: 
    asyncio.run(run_workflow())
finally:
    #wait_for_all_tracers()
    print("Tracers are done")
"""