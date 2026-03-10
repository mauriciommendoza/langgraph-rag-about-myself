"""
Grader Chains Module.
Contains prompt templates and Chains used to grade document relevance, 
hallucinations, and answer usefulness.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

from src.config.llm import llm

def build_retrieval_grader() -> Runnable:
    """
    Builds a chain that assesses the relevance of a retrieved document to a user's question.
    
    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "document"],
    )
    return prompt | llm.with_structured_output(method="json_mode")


def build_hallucination_grader() -> Runnable:
    """
    Builds a chain that checks if a generated answer is hallucinated or grounded in the facts.
    
    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in a set of facts. 
        If the answer contains information not present in the facts, or goes beyond what the facts say, grade it as 'no'. 
        If the answer strictly uses information from the facts, grade it as 'yes'. 
        Your response must be a valid JSON object with a single key 'score' and value 'yes' or 'no'. 
        Do not include any explanation, commentary, or additional keys.

        Facts:
        {documents}

        Answer:
        {generation}

        Respond with:
        {{"score": "yes"}} or {{"score": "no"}} only.
        """,
        input_variables=["generation", "documents"]
    )
    return prompt | llm.with_structured_output(method="json_mode")


def build_answer_grader() -> Runnable:
    """
    Builds a chain that determines if the generated answer actually resolves the user's question.
    
    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. 
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         
        Here is the answer:
        {generation} 

        Here is the question: {question}
        """,
        input_variables=["generation", "question"],
    )
    return prompt | llm.with_structured_output(method="json_mode")

# Export instantiated chains
retrieval_grader = build_retrieval_grader()
hallucination_grader = build_hallucination_grader()
answer_grader = build_answer_grader()
