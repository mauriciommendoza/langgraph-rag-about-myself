"""
Centralized Prompt Registry.
All LLM prompt templates are defined here for easy versioning and maintenance.
Edit prompts in this single file to iterate on prompt engineering.
"""

from langchain_core.prompts import PromptTemplate

# --- RAG Generation ---
RAG_GENERATION_PROMPT = PromptTemplate(
    template=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    ),
    input_variables=["question", "context"],
)

# --- Retrieval Grader ---
RETRIEVAL_GRADER_PROMPT = PromptTemplate(
    template=(
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Here is the retrieved document:\n\n{document}\n\n"
        "Here is the user question: {question}\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.\n"
        "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."
    ),
    input_variables=["question", "document"],
)

# --- Hallucination Grader ---
HALLUCINATION_GRADER_PROMPT = PromptTemplate(
    template=(
        "You are a grader assessing whether an answer is grounded in a set of facts. "
        "If the answer contains information not present in the facts, or goes beyond what the facts say, grade it as 'no'. "
        "If the answer strictly uses information from the facts, grade it as 'yes'. "
        "Your response must be a valid JSON object with a single key 'score' and value 'yes' or 'no'. "
        "Do not include any explanation, commentary, or additional keys.\n\n"
        "Facts:\n{documents}\n\n"
        "Answer:\n{generation}\n\n"
        'Respond with: {{"score": "yes"}} or {{"score": "no"}} only.'
    ),
    input_variables=["generation", "documents"],
)

# --- Answer Grader ---
ANSWER_GRADER_PROMPT = PromptTemplate(
    template=(
        "You are a grader assessing whether an answer is useful to resolve a question. "
        "Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. "
        "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n\n"
        "Here is the answer:\n{generation}\n\n"
        "Here is the question: {question}"
    ),
    input_variables=["generation", "question"],
)
