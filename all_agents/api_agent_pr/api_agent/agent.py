import os, pprint


from typing import Annotated, List, Tuple, TypedDict
import operator
from langchain_core.pydantic_v1 import BaseModel, Field
#from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_community.agent_toolkits.openapi import API_PLANNER_PROMPT
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing import Union, Literal
from langchain import hub
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec, ReducedOpenAPISpec
import yaml, re
from langchain_core.tools import BaseTool, Tool
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.planner import RequestsGetToolWithParsing
from langchain.chains.llm import LLMChain
from langchain_community.agent_toolkits.openapi.planner_prompt import PARSING_GET_PROMPT

#from dotenv import load_dotenv
#load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"
ACCESS_TOKEN="b9d1989f8e5b2855139a84710e6350082bbe8757"

###########################################
### Classes and Utils

class PlanExecute(TypedDict):
    input: str
    openAPIspec: ReducedOpenAPISpec
    plan: List[str]
    past_steps: Annotated[List[List], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

def auth_headers():
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

def get_tools(task):
    
    prompt_get = """
        
        Your task is to extract some information according to these instructions: {instructions}
        When working with API objects, you should usually use ids over names.
        If the response indicates an error, you should instead output a summary of the error.

        Output:
        """.format(instructions=task).replace("{", "{{").replace("}", "}}")
    tool_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt_get),("placeholder", "{messages}"),]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_chain = tool_prompt | llm
    headers = auth_headers()
    requests_wrapper = RequestsWrapper(headers=headers)
    tools : List[BaseTool] =[]
    tools.append(
        RequestsGetToolWithParsing(
            requests_wrapper=requests_wrapper,
            llm_chain = llm_chain,
            allow_dangerous_requests=True,
            allowed_operations=["GET"],)
    ) 
    #print(f"Tools ------------- {tools}")
    return tools

def get_planner_agent(openAPIspec):
    # look through the endpoints descriptions to find the right APIs to call
    endpoint_descriptions = [
        f"{name} {description[:20]}" if description is not None else "" for name, description, _ in openAPIspec.endpoints
    ]
    endpoints =  "- ".join(endpoint_descriptions)
    sysprompt = """
            You are an agent that assists with user queries against API, things like querying information or creating resources.
            Some user queries can be resolved in a single API call, particularly if you can find appropriate params from the OpenAPI spec; though some require several API calls.
            You should always plan your API calls first, and then execute the plan second.
            If the plan includes a DELETE call, be sure to ask the User for authorization first unless the User has specifically asked to delete something.
            You should never return information without executing the api_controller tool.

            Each plan can be used to generate the right API calls to assist with a user query. 
            You should always have a plan before trying to call the API controller.

            api_controller: Can be used to execute a plan of API calls, like api_controller(plan).
                                                            
            The API endpoints are {endpoints}.

            For the given objective, come up with a simple step by step plan. 
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            """.format(endpoints=endpoints).replace("{", "{{").replace("}", "}}")
    #print(sysprompt)
    planner_prompt = ChatPromptTemplate.from_messages(
        [("system", sysprompt),("placeholder", "{messages}"),]
    )
    print("Getting planner agent")
    #print(planner_prompt)
    planner = planner_prompt | ChatOpenAI(
                model="gpt-4o", temperature=0
            ).with_structured_output(Plan)
    return planner

def get_orchestrator():
    """
    Agent that assists with user queries against API, things like querying information or creating resources.
    Some user queries can be resolved in a single API call, particularly if we can find appropriate params from the OpenAPI spec; 
    though some require several API calls.
    Always plan your API calls first, and then execute the plan second.
    """
    return

def get_api_docs(task, api_spec):
    """
    get the relevant API documentation for the task

    """
    pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
    matches = re.findall(pattern, task)
    endpoint_names = [
        "{method} {route}".format(method=method, route=route.split("?")[0])
        for method, route in matches
    ]
    docs_str = ""
    for endpoint_name in endpoint_names:
        found_match = False
        for name, _, docs in api_spec.endpoints:
            regex_name = re.compile(re.sub(r"\{.*?\}", ".*", name))
            if regex_name.match(endpoint_name):
                found_match = True
                docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"
        if not found_match:
            raise ValueError(f"{endpoint_name} endpoint does not exist.")

    #print(docs_str)
    return docs_str
    
def get_execution_agent():
    #TODO: NOT USED
    tools = get_tools()
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


############################################
### Tools

### Nodes
async def create_plan(state: PlanExecute):
    """Come up with a simple step by step plan given the initial message
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        plan (dict): A list of steps to follow 
    """
    with open("./openapispecs/catalog/catalog_view.yaml") as f:
        raw_openapi_spec = yaml.load(f, Loader=yaml.Loader)
    openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=False)
    planner_agent = get_planner_agent(openapi_spec)
    plan = await planner_agent.ainvoke({"messages": [("user", state["input"])]})
    #print(f"Here is the plan: {plan}")
    return {"plan": plan.steps, "openAPIspec": openapi_spec}

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
    #TODO: add tools to perform GET, POST, etc. requests
    tools = get_tools(task)
    api_url = "https://useast.api.elasticpath.com"
    openAPIspec = state["openAPIspec"]
    api_docs = get_api_docs(task, openAPIspec)
    prompt = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
            If you cannot complete them and run into issues, you should explain the issue. 
            If you're unable to resolve an API call, you can retry the API call. 
            When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.

            Here is documentation on the API:
            Base url: {api_url}
            Endpoints:
            {api_docs}


            Here are tools to execute requests against the API: RequestsGetToolWithParsing. 
            Make sure that  you pass a valid JSON object to RequestGetToolWithParsing.
            
            """.format(api_url=api_url, api_docs=api_docs).replace("{", "{{").replace("}", "}}")
    #print(prompt)
    exec_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt),("placeholder", "{messages}"),]
    )
    llm = ChatOpenAI(model="gpt-4o")
    agent_executor = create_react_agent(llm, tools, state_modifier=exec_prompt)
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


config = {"recursion_limit": 10}
input = {"input": "show me all the product for the node named running"}

import asyncio
async def run_workflow():
    async for output in graph.astream(input, config=config):
        for key, value in output.items():
            # Node
            print(f"\033[93mOutput from node '{key}':\033[0m")
            if key == "planner" and value is not None:
                pprint.pprint( str(value)[:100])
            else:
                pprint.pprint(value)
            # Optional: print full state at each node
            #if key != "__end__":
            #    print(f"\033[92m{value}\033[0m")
        print("\n---\n")

try: 
    asyncio.run(run_workflow())
finally:
    #wait_for_all_tracers()
    print("Tracers are done")
