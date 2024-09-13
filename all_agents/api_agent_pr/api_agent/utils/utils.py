import requests, yaml, re
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec, ReducedOpenAPISpec
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Find a match for the endpoint 
def find_match_for_endpoint(input_string):
    
    input = input_string
    endpoints = get_API_names_description()
    
    sys_prompt_str = """
        Agent that given an {input} can find the most appropriate API endpoint to answer the plan. 
        You can only use these API endpoints {apiendpoints} descriptions to find the right one.
        Your answer should be in the format of "GET /pcm/products/{{product_id}}" or "POST /products" and not include the description.
    """
    
    sys_prompt_tpl = ChatPromptTemplate([
        ("system", sys_prompt_str), ("user", input)
        ], 
    )
    sys_prompt = sys_prompt_tpl.invoke(
        {"input": input, "apiendpoints": endpoints}
    )
    matcher = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = matcher.invoke( sys_prompt)
    return response.content


def get_API_names_description():
    """
    Returns a list of API names and descriptions.
    """
    #url = "https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/catalog/catalog_view.yaml"
    url = "https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/pim/pim.yaml"
    response = requests.get(url)
    if response.status_code == 200:
        raw_openapi_spec = yaml.safe_load(response.text)
    else:
        raise ValueError(f"Failed to fetch OpenAPI spec: {response.text}")

    #with open("./openapispecs/catalog/catalog_view.yaml") as f:
    #    raw_openapi_spec = yaml.load(f, Loader=yaml.Loader)
    openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=False)
    endpoint_descriptions = [
        f"{name} {description[:1000]}" if description is not None else "" for name, description, _ in openapi_spec.endpoints
    ]
    #endpoints =  "\n ".join(endpoint_descriptions)
    
    return endpoint_descriptions

# Get the right spec for the endpoint
def get_OpenAPI_spec_for_endpoint(endpoint: str):
    """
    Returns the OpenAPI spec for the specified endpoint.
    It expects an input like "GET /products/{product_id}"
    """
    #url = "https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/catalog/catalog_view.yaml"
    url = "https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/pim/pim.yaml"
    response = requests.get(url)
    if response.status_code == 200:
        raw_openapi_spec = yaml.safe_load(response.text)

    openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=False)
    # TODO: remember that I had to change the dereference function and added skip_keys=["examples"]
    pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
    matches = re.findall(pattern, endpoint)
    endpoint_names = [
        "{method} {route}".format(method=method, route=route.split("?")[0])
        for method, route in matches
    ]
    docs_str = ""
    for endpoint_name in endpoint_names:
        found_match = False
        for name, _, docs in openapi_spec.endpoints:
            #TODO: improve regex to match exactly the endpoint name
            regex_name = re.compile(re.sub(r"\{.*?\}", ".*", name))
            #regex_name = re.compile(f"^{re.sub(r'\{.*?\}', '.*', name)}$")
            if regex_name.match(endpoint_name):
                found_match = True
                docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"
        if not found_match:
            raise ValueError(f"{endpoint_name} endpoint does not exist.")
    return docs_str