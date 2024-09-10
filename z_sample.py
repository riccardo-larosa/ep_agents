from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
#from tool1 import MockSearchTool, MockCalcTool
from langgraph.prebuilt import ToolNode, tools_condition
#from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits.openapi.planner import _create_api_controller_tool, _create_api_planner_tool

class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    question_type: str


graph_builder = StateGraph(AssistantState)
from langchain_core.tools import tool

#@tool
#def MockSearchTool(input):
#    """"
#    #mock function to search
#    """
#    return "MockSearchTool"
#@tool
#def MockCalcTool(number):
#    """
#    #mock calculation tool
#    """
#    return 99
#tool1 = MockSearchTool
#tool2 = MockCalcTool
#tools = [tool1, tool2]

from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
import yaml
with open("./openapispecs/files/files.yaml") as f:
    raw_openapi_spec = yaml.load(f, Loader=yaml.Loader)
openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=True)
from langchain_community.utilities.requests import RequestsWrapper
ACCESS_TOKEN="05021a80cf0919535a7b425ef33ffc7a218ab365"
def auth_headers():
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

headers = auth_headers()
requests_wrapper = RequestsWrapper(headers=headers)
llm = ChatOpenAI(model="gpt-4o")
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
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools=tools)

def chatbot(state: AssistantState):
    print(state["messages"])

    # Define your prompt template
    template = """You are an agent that assists with user queries against API, things like querying information or creating resources.
    Some user queries can be resolved in a single API call, particularly if you can find appropriate params from the OpenAPI spec; though some require several API calls.
    You should always plan your API calls first, and then execute the plan second.
    If the plan includes a DELETE call, be sure to ask the User for authorization first unless the User has specifically asked to delete something.
    You should never return information without executing the api_controller tool.


    Here are the tools to plan and execute API requests: api_planner: Can be used to generate the right API calls to assist with a user query, like api_planner(query). Should always be called before trying to call the API controller.
    api_controller: Can be used to execute a plan of API calls, like api_controller(plan).


    Starting below, you should follow this format:

    User query: the query a User wants help with related to the API
    Thought: you should always think about what to do
    Action: the action to take, should be one of the tools [api_planner, api_controller]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create
    Final Answer: the final output from executing the plan


    Example:
    User query: can you add some trendy stuff to my shopping cart.
    Thought: I should plan API calls first.
    Action: api_planner
    Action Input: I need to find the right API calls to add trendy items to the users shopping cart
    Observation: 1) GET /items with params 'trending' is 'True' to get trending item ids
    2) GET /user to get user
    3) POST /cart to post the trending items to the user's cart
    Thought: I'm ready to execute the API calls.
    Action: api_controller
    Action Input: 1) GET /items params 'trending' is 'True' to get trending item ids
    2) GET /user to get user
    3) POST /cart to post the trending items to the user's cart
    ...

    Begin!

    Conversation history:
    {history}

    Latest human message: {input}

    AI assistant response:"""

    prompt = PromptTemplate(
        input_variables=["history", "latest_message"], template=template
    )

    # Create the chain
    chain = prompt | llm_with_tools

    return {
        "messages": [
            chain.invoke({"input": state["messages"][-1], "history": state["messages"]})
        ]
    }

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# graph.add_edge("then_something", END)
#memory = SqliteSaver.from_conn_string(":memory:")


graph = graph_builder.compile()

config = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}, config):
        for value in event.values():
            #snapshot = graph.get_state(config)
            # print("Next Node:", snapshot.next[0])
            if value["messages"][-1].content:
                print("Assistant:", value["messages"][-1].content)