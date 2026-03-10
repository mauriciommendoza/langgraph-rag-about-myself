"""
Routing Module.
Used to route an incoming question either to the vectorstore or to a web search.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

from src.config.llm import llm

def build_question_router() -> Runnable:
    """
    Builds a chain that decides the data source for a given question.
    
    Returns:
        Runnable: A chain outputting JSON with a 'datasource' key ('web_search' or 'vectorstore').
    """
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. 
        Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        Return the a JSON with a single key 'datasource' and no premable or explaination. 
        
        Question to route: 
        {question}""",
        input_variables=["question"],
    )
    return prompt | llm.with_structured_output(method="json_mode")

question_router = build_question_router()
