from api_agent.utils.utils import get_API_names_description, get_OpenAPI_spec_for_endpoint, find_match_for_endpoint
from api_agent.utils.tools import exec_put_request


cart_endpoint='/v2/carts'
cart_body= "{\"data\":{\"name\":\"my_cart\",\"description\":\"A cart for testing\", \"discount_settings\": {\"custom_discounts_enabled\": true}}}"
pcm_endpoint="/pcm/products/c72866b2-3979-4be3-8d84-9c5e2ce8948e"
#pcm_body= "{\"data\":{\"type\":\"product\",\"attributes\": {\"name\": \"my product\", \"sku\": \"my_product\", \"commodity_type\": \"physical\"}}}"
#pcm_body = "{\"data\":{\"id\":\"c72866b2-3979-4be3-8d84-9c5e2ce8948e\",\"type\":\"product\",\"attributes\":{\"status\":\"live\"}}}"
pcm_body = " {'data': {'type': 'product', 'id': 'c72866b2-3979-4be3-8d84-9c5e2ce8948e', 'attributes': {'status': 'draft'}}}"
token= '547f978e0fb4df4ea6ce3d3dc8d646824761127e'
#response = requests.post("https://useast.api.elasticpath.com" + cart_endpoint, headers=create_headers(token), data=cart_body)
response = exec_put_request(pcm_endpoint, token=token, data=pcm_body)

#find the appropriate API endpoint 
#api_endpoint = find_match_for_endpoint("add product to cart")
#get the full API spec for this endpoint
#api_spec = get_OpenAPI_spec_for_endpoint(api_endpoint)
#print(api_spec)

#endpoints = get_API_names_description()
#for endpoint in endpoints:
#    print(endpoint[:170])
#print(endpoints)

#openapispec = get_OpenAPI_spec_for_endpoint("https://raw.githubusercontent.com/elasticpath/elasticpath-dev/main/openapispecs/currencies/OpenAPISpec.yaml|PUT /v2/currencies/{currencyID}")
#print(openapispec)