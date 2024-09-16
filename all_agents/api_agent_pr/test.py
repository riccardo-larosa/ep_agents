from api_agent.utils.utils import get_API_names_description, get_OpenAPI_spec_for_endpoint

endpoints = get_API_names_description()
for endpoint in endpoints:
    print(endpoint[:170])
#print(endpoints)

openapispec = get_OpenAPI_spec_for_endpoint("https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/currencies/OpenAPISpec.yaml|PUT /v2/currencies/{currencyID}")
print(openapispec)