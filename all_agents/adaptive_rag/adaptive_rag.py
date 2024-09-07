import os, pprint
#from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


#load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="ep_agents"
os.environ["LANGCHAIN_TRACING_V2"]="true"

#############################################
### Router
# Data Model
class CM_vectorsearch(BaseModel):
    """
    The internet. Use CM_vectorsearch for questions that are related to anything else 
    than agents, prompt engineering, and adversarial attacks.
    """

    query: str = Field(description="The query to use when searching the internet.")


class API_vectorsearch(BaseModel):
    """
    A API_vectorsearch containing documents related to agents, prompt engineering, and adversarial attacks. 
    Use the API_vectorsearch for questions on these topics.
    """

    query: str = Field(description="The query to use when searching the API_vectorsearch.")


# Preamble
preamble = """You are an expert at routing a user question to a vectorstore or web search.
The API_vectorsearch contains documents related to agents, prompt engineering, and adversarial attacks.
Use the API_vectorsearch for questions on these topics. Otherwise, use CM_vectorsearch."""

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble),
        ("human", "{question}"),
    ]
)

# LLM with tool use 
llm = ChatOpenAI(model="gpt-4o", temperature=0, )
structured_llm_router = llm.bind_tools(
    tools=[CM_vectorsearch, API_vectorsearch]
)

question_router = route_prompt | structured_llm_router
"""
response = question_router.invoke(
    {"question": "Who will the Bears draft first in the NFL draft?"}
)
print(response.tool_calls)
response = question_router.invoke({"question": "What are the types of agent memory?"})
print(response.tool_calls)
response = question_router.invoke({"question": "Hi how are you?"})
#print("tool_calls" in response.response_metadata)
print(response.tool_calls)
"""
#############################################
### Retrieval Grader
# Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Prompt
preamble_retriever = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble_retriever),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

question = "is agent memory used in LangChain?"
#docs = retriever.invoke(question)
from langchain.schema import Document
docs = [Document(page_content="In AI systems like LangChain, agent memory is used to store and manage information during an interaction."),
        Document(page_content="There are three types of agent memory: sensory memory, short-term memory, and long-term memory")]



# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

retrieval_grader = grade_prompt | structured_llm_grader
doc_txt = docs[1].page_content
#doc_txt = docs[1]
#response = retrieval_grader.invoke({"question": question, "document": doc_txt})
#print(response)


#############################################
### Generate
# Preamble
preamble_generate = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise."""

# Prompt
def prompt(x):
    #print(f"Question: {x['question']} \nAnswer: documents: {x["documents"]} ")
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=preamble_generate),
            HumanMessage(
                f"Question: {x['question']} \nAnswer: ",
                additional_kwargs={"documents": x["documents"]},
            )
        ]
    )


# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
#generation = rag_chain.invoke({"documents": docs, "question": question})
#print(rag_chain)
#print(generation)

#############################################
### Hallucination Grader
class GradeRelevancy(BaseModel):
    """Binary score for relevance check on hallucinated text."""

    binary_score: str = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )

# Preamble
preamble_grader = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    Give a binary score 'yes' or 'no'. 
    'Yes' means that the answer is grounded in / supported by the set of facts."""

# Prompt
relevancy_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble_grader),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(
    GradeRelevancy, 
)

relevancy_grader = relevancy_prompt | structured_llm_grader
#result = relevancy_grader.invoke({"documents": docs, "generation": generation})
#print(result)

#############################################
### Answer Grader
class GradeAnswer(BaseModel):
    """Binary score for correctness check on generated answer."""

    binary_score: str = Field(
        description="Answer is correct, 'yes' or 'no'"
    )

# Preamble
preamble_answer_grader = """You are a grader assessing whether an LLM generation is correct. \n
    Give a binary score 'yes' or 'no'. 
    'Yes' means that the answer is correct."""

# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(
    GradeAnswer, 
)

# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble_answer_grader),
        ("human", "Correct answer: \n\n {answer} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

#result = answer_grader.invoke({"answer": "Agent memory is used in LangChain", "generation": generation})
#print(result)

#############################################
### Graph
from typing import TypedDict, Literal, List
from typing_extensions import TypedDict
from langchain.schema import Document

class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

#############################################
### Nodes

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve documents from CM or API engine based on the question.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, documents, that contains the retrieved documents.
    """
    print("----> retrieve_documents")
    question = state["question"]
    print(f"Question: {question}")

    # Retrieval
    #TODO: Implement retrieval
    documents = [Document(page_content= "In AI systems like LangChain, agent memory is used to store and manage information during an interaction. There are different types of agent memory, including: short-term memory, long-term memory, conversation buffer memory, knowledge base memory, and vector-based memory. Each type serves a specific purpose in helping the agent understand and respond to user queries.")  ]

    return {"documents": documents, "question": question}

def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer to the question based on the retrieved documents.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, generation, that contains the generated answer.
    """
    print("----> generate_answer")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # Generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: GraphState) -> GraphState:
    """Grade the relevance of the retrieved documents to the question.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): Updates docuements key with only filtered relevant documents.
    """
    print("----> grade_documents")
    question = state["question"]
    documents = state["documents"]

    # Grading
    relevant_docs = []
    for doc in documents:
        response = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if response.binary_score == "yes":
            print("--> Relevant document")
            relevant_docs.append(doc)
        else:
            print("--> Irrelevant document")
            continue
    return {"documents": relevant_docs, "question": question}

#############################################
### Edges

def route_question(state):
    """Route the question to the appropriate engine based on the question type.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, documents, that contains the retrieved documents.
    """
    print("----> route_question")
    question = state["question"]
    print(f"Question: {question}")
    source = question_router.invoke({"question": question}) 
    print(f"Source: {source}")
    if "tool_calls" not in source.additional_kwargs:
        print("---no tools called---")
        raise "Router could not decide source"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source"
    
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "API_vectorsearch": 
        print("--> API search")
        return "API_vectorsearch"
    elif datasource == "CM_vectorsearch":
        print("--> CM search")
        return "CM_vectorsearch"
    else:
        raise "Router could not determine source"

def decide_to_generate(state):
    """Decide whether to generate an answer based on the retrieved documents, or re-generate a question
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        str: Binary decision for next node to call
    """
    print("----> decide_to_generate")
    print("-- Assess Graded Documents --")
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("All documents have been filtered out by the grader. We can't answer the question")
        print("--> No documents found")
        return "no documents found"
    else:
        # we have relevant documents, generate an answer
        print("--> Documents found")
        return "generate_answer"
    
def grade_generation_vs_docs_and_question(state):
    """Grade the relevancy of the generated answer from the question and retrieved documents.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        str: Decision for next node to call
    """
    print("----> grade_generation_vs_docs_and_question")
    print("-- Assess Graded Generation (or hallucinations)--")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    response = relevancy_grader.invoke({"documents": documents, "generation": generation})
    if response.binary_score == "yes":
        print("--> Generation is relevant to the documents")
        # Check question-answer relevancy TODO: check the inputs
        #score = answer_grader.invoke({"answer": question, "generation": generation})
        #grade = score.binary_score
        grade = "yes"
        if grade == "yes":
            print("--> Generation addresses question")
            return "useful"
        else:
            print("--> Generation does not address question")
            return "notuseful"
    else:
        print("--> Generation is NOT relevant to the documents re-try")
        return "not supported"
    
#############################################
### Graph
import pprint
from langgraph.graph import StateGraph, END, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)

# Define the edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "CM_vectorsearch": "retrieve_documents",
        "API_vectorsearch": "retrieve_documents",
    },
)
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate_answer": "generate_answer",
        "no documents found": END,
    },
)
workflow.add_conditional_edges(
    "generate_answer",
    grade_generation_vs_docs_and_question,
    {
        "useful": END,
        "notuseful": END, # we would need a fallback here
        "not supported": "retrieve_documents",
    },
)

graph = workflow.compile()
"""
input = {"question": "is agent memory used in LangChain?"}
for output in graph.stream(input):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node

    print("\n---\n")

pprint.pprint(value["generation"])
"""