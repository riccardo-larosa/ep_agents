
from agentic_rag.agent import graph

import os, pprint
os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"
inputs = {
    "messages": [
        ("user", "what is the right apis for pxm "),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")