import requests, os

# Your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Set the headers with your API key
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the API endpoint (this could be any valid OpenAI endpoint, e.g., the Completions endpoint)
#url = "https://api.openai.com/v1/models"
url = "https://api.openai.com/v1/embeddings"

# Send a GET request to the OpenAI API
response = requests.post(url, headers=headers, 
                         data='{"input": "Once upon a time, there was a frog.",\
                                "model": "text-davinci-003"}')
print(response)
# Extract the headers from the response
response_headers = response.headers

print(response_headers)
# Print the quota-related headers
# Check for rate-limiting headers like X-Ratelimit-Limit, X-Ratelimit-Remaining, etc.
for header, value in response_headers.items():
    if "ratelimit" in header.lower():
        print(f"{header}: {value}")