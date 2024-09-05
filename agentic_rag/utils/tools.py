from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def getEPAPIsResults():
    """Use this to find the right APIs endpoint for Elastic Path."""
    #TODO: Add logic from chatbot app
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db_name = "rag_db"        
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"
    # Create a MongoDBAtlasVectorSearch object
    db = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        embeddings,
        index_name = vector_search_index
    )
    # make sure that prompt has all the messages
    results = db.similarity_search_with_score(prompt, k=TOP_K)
    """
    return "GET https://api.elasticpath.dev/api"

@tool
def getEPCMResults():
    """Use this to find the right information from Commerce Manager for Elastic Path queries that don't require an API.
        This tool is geared towards users who are looking for information on how to use Commerce Manager.
    """
    return "EP Commerce Manager is a great tool that can help you find the right information."

tools = [TavilySearchResults(max_results=1), getEPAPIsResults, getEPCMResults]