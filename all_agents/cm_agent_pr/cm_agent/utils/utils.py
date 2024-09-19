import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

MODEL_RETRIEVAL =  "gpt-4o-mini" #"gpt-3.5-turbo"
MODEL_GENERATION = "gpt-4o"
MODEL_TO_USE = "OPENAI" #"COHERE"

#############################################
### Retrieval Grader
# Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def get_CM_docs(question):
    """get retriever to query Elastic Path documentation for Commerce Manager"""
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db_name = "rag_db"
    collection_name = "epdocs_openaiembeddings"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"
    # Create a MongoDBAtlasVectorSearch object
    db = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        embeddings,
        index_name = vector_search_index
    )
    # TODO: I could 'pre-filter' the query if I know I want something specific from the docs 
    results = db.similarity_search_with_score(question, k=5)
    return results

def get_retrieval_grader():
    # LLM with function call
    if MODEL_TO_USE == "OPENAI":    
        llm = ChatOpenAI(model=MODEL_RETRIEVAL, temperature=0)
    else:
        llm = ChatCohere(model="command-r", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

