"""
Response Generator Module.
Contains the RAG prompt and chain responsible for generating the final answer.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.config.llm import llm

def build_rag_chain() -> Runnable:
    """
    Builds the Retrieval-Augmented Generation (RAG) chain.
    
    Returns:
        Runnable: A chain that takes a context and question, and returns an answer string.
    """
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context"],
    )
    return prompt | llm | StrOutputParser()

rag_chain = build_rag_chain()
