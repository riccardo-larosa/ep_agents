from cm_agent.utils.state import GraphState
from cm_agent.utils.utils import get_CM_docs, GradeDocuments, MODEL_TO_USE, MODEL_RETRIEVAL, MODEL_GENERATION
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



#############################################
### Nodes

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve documents from CM or API engine based on the question.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, documents, that contains the retrieved documents.
    """
    question = state["question"] # this is the input
    documents = get_CM_docs(state["question"])
    state["documents"] = [doc.page_content for doc, _score in documents]
    return {"documents": documents, "question": question}

def grade_retrieved_documents(state: GraphState) -> GraphState:
    """Grade the relevance of the retrieved documents to the user question.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, documents, that contains the retrieved documents.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
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

    # Score each doc
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc[0].page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def generate_response(state: GraphState) -> GraphState:
    """Generate a response based on the retrieved documents and the user question.
    
    Args:
        state (dict): The current state of the graph.
    Returns: 
        state (dict): New key added to state, generation, that contains the generated response.
    """
    question = state["question"]
    documents = state["documents"]
    PROMPT_TEMPLATE = """
        \n\n\033[33m--------------------------\033[0m\n\n
        You are knowledgeable about Elastic Path products. You can answer any questions about 
        Commerce Manager, 
        Product Experience Manager also known as PXM,
        Cart and Checkout,
        Promotions,
        Composer,
        Payments
        Subscriptions,
        Studio.
        {prompt_base}
        Answer the question based only on the following context:
        \n\033[33m--------------------------\033[0m\n
        {context}
        \n\033[33m--------------------------\033[0m\n
        Answer the following question based on the above context: {question}
        \n\033[33m--------------------------\033[0m\n
        """
    PROMPT_BASE = """
        Build any of the relative links using https://elasticpath.dev as the root
        """
    context_text = "\n\n\033[32m---------------\033[0m\n\n".join([doc.page_content for doc, _score in documents])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_context = prompt_template.format(prompt_base=PROMPT_BASE, context=context_text, question=question)
    if MODEL_TO_USE == "OPENAI":    
        model = ChatOpenAI(model=MODEL_GENERATION, temperature=0.7)
    else:
        model = ChatCohere(model="command-r", temperature=0.7)
    #model = ChatOpenAI(temperature=0.7, model=MODEL_GENERATION)
    response = model.invoke( prompt_context)
    #rag_chain = prompt_context | model | StrOutputParser()
    #generation = rag_chain.invoke({"context": documents, "question": question})
    generation = response.content
    return {"documents": documents, "question": question, "generation": generation}


def rewrite_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---REWRITE QUERY---")
    question = state["question"]
    if MODEL_TO_USE == "OPENAI":    
        model = ChatOpenAI(model=MODEL_RETRIEVAL, temperature=0.7)
    else:
        model = ChatCohere(model="command-r", temperature=0.7)
    #model = ChatOpenAI(model="gpt-4o", temperature=0)
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | model | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
   
    return {"question": better_question}

##############################################
### Edges

def decide_to_rewrite_query(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"