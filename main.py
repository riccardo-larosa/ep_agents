
#from all_agents.agentic_rag.agent import graph
from all_agents.cm_agent_pr.cm_agent.main import graph
#from all_agents.api_agent.agent import get_CM_answer
#from all_agents.api_agent.agent import get_planner_agent

import os, pprint
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"
ACCESS_TOKEN="05021a80cf0919535a7b425ef33ffc7a218ab365"
inputs = {
    "question":  "what is a pricebook "
}

# Get the answer from the CM agent
#answer = get_CM_answer("what is a node")
#print(answer)

#
#for output in graph.stream(inputs):
#    for key, value in output.items():
#        pprint.pprint(f"Output from node '{key}':")
#        pprint.pprint("---")
#        pprint.pprint(value, indent=2, width=160, depth=None)
#    pprint.pprint("\n---\n")

from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
#from langchain_community.agent_toolkits.openapi import API_PLANNER_PROMPT
import langchain_community.agent_toolkits.openapi.planner_prompt as prompts
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
import yaml
#from langchain_core.tools import BaseTool, Tool
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

with open("./openapispecs/files/files.yaml") as f:
    raw_openapi_spec = yaml.load(f, Loader=yaml.Loader)
openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=True)
#print(openapi_spec)
input = "show me all the files"

@tool
def api_planner(input ):
    """
    Can be used to generate the right API calls to assist with a user query, like api_planner(query). 
    Should always be called before trying to call the API controller.

    """
    print("api_planner -------------------------------")
    llm = ChatCohere(model="command-r", temperature=0)

    endpoint_descriptions = [
            f"{name} {description}" for name, description, _ in openapi_spec.endpoints
        ]
    endpoints = {"endpoints": "- " + "- ".join(endpoint_descriptions)}
    planner_prompt = PromptTemplate(
        template=prompts.API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables=endpoints,
    )
    
    planner = planner_prompt | llm
    #tool = Tool(
    #    name=prompts.API_PLANNER_TOOL_NAME,
    #    description=prompts.API_PLANNER_TOOL_DESCRIPTION,
    #    #func=planner.ainvoke,
    #    )
    print(f"query: {input}")
    results=planner.invoke({"query": input})
    print(f"api_planner results: {results}")
    return results 

@tool
def api_controller(action, openapi_spec=openapi_spec):
    """
    Can be used to call the right API endpoint to assist with a user query, like api_controller(action). 
    Should always be called after calling the API planner.

    """
    llm = ChatCohere(model="command-r", temperature=0)
    print(f"action: {action}")
    #results=llm.invoke({"input": action})
    results = {"response":"CURL GET /files"}
    print(f"api_controller results: {results}")
    return results

#tools = [api_planner, api_controller]
#print(tools)
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, List, Tuple, TypedDict, Literal, Optional
import operator
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.openapi.planner import _create_api_controller_tool, _create_api_planner_tool

#tool_node = ToolNode(tools=tools)
###
class OrchestratorPlanSequence(BaseModel):
    action: Literal['api_planner', 'api_controller']
    action_input: Optional[str] = Field(description="this is called 'Action Input' in the prompt")
    observation: Optional[str]
    thought: Optional[str]
    

class OrchestratorPlan(BaseModel):
    plan: Optional[str]
    orchestration_sequence: List[OrchestratorPlanSequence]
    final_answer: Optional[str]
 

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[List], operator.add]
    response: str

from langchain_community.utilities.requests import RequestsWrapper
def auth_headers():
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

def run_orchestrator( input):
    #llm = ChatCohere(model="command-r", temperature=0)
    parser = PydanticOutputParser(pydantic_object=OrchestratorPlan)
    llm = ChatOpenAI(model="gpt-4o")
    
    
    headers = auth_headers()
    requests_wrapper = RequestsWrapper(headers=headers)


    tools = [_create_api_planner_tool(openapi_spec, llm),
             _create_api_controller_tool(
            openapi_spec,
            requests_wrapper,
            llm,
            allow_dangerous_requests=True,
            allowed_operations=["GET", "POST"],
            ), ]
    tool_names = ", ".join([tool.name for tool in tools])
    tool_descriptions = "\n ".join([f"{tool.name}: {tool.description}" for tool in tools])
    vars = {
                "tool_names": tool_names,
                "tool_descriptions": tool_descriptions
                ,
                #"format_instructions": parser.get_format_instructions(),
            }
    print(vars)
    orchestrator_prompt = PromptTemplate(
        template=prompts.API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables=vars,
    )

    system_message = orchestrator_prompt.template
    system_message = system_message.replace("{input}", "show me all the files").replace("{agent_scratchpad}", "{}")
    system_message = system_message.replace("{tool_names}", tool_names).replace("{tool_descriptions}", tool_descriptions)
    print(system_message)
    #orchestrator = orchestrator_prompt | llm.bind_tools(tools) #| parser#.with_structured_output(OrchestratorPlan)#.bind_tools(tools)
    #results=orchestrator.invoke({"input": input, "agent_scratchpad": {}}).tool_calls
    app = create_react_agent(llm, tools, state_modifier=system_message)

    
    messages = app.invoke({"messages": [("human", "show me all the files")]})
    print({
        "input": input,
        "output": messages["messages"][-1].content,
    })
    #print(results)
    return 

from langgraph.graph import StateGraph, START, END
#from typing_extensions import TypedDict, Annotated



#graph = StateGraph(
#    PlanExecute
#)

#graph.add_node("orchestrator", run_orchestrator)
#graph.add_node("tools", tool_node)
#graph.set_entry_point("orchestrator")
#graph.add_edge("tools", "orchestrator")
#graph.add_conditional_edges("orchestrator",  tools_condition)
#graph.add_edge("something", END)
#app = graph.compile()

inputs = {
    "input":  "show me all the files",
    "agent_scratchpad": {}
}
#for output in app.stream(inputs):
#    for key, value in output.items():
#        pprint.pprint(f"Output from node '{key}':")
#        pprint.pprint("---")
#        pprint.pprint(value, indent=2, width=160, depth=None)
#    pprint.pprint("\n---\n")

orch = run_orchestrator(inputs)
print(orch)
#planner = api_planner("show me all the files")



