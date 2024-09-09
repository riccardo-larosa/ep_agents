
#from all_agents.agentic_rag.agent import graph
from all_agents.cm_agent_pr.cm_agent.main import graph
#from all_agents.api_agent.agent import get_CM_answer

import os, pprint
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"
inputs = {
    "question":  "what is a pricebook "
}

# Get the answer from the CM agent
#answer = get_CM_answer("what is a node")
#print(answer)

for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=160, depth=None)
    pprint.pprint("\n---\n")