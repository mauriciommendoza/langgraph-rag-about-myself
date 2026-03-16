"""
Chainlit UI Application.
Provides an interactive web-based chat interface for the LangGraph RAG Agent.
Run with: chainlit run app_ui.py --port 8080
"""

import asyncio
import chainlit as cl
from src.graph.builder import app


def run_graph(question: str) -> list:
    """
    Runs the LangGraph agent synchronously and collects all node outputs.

    Args:
        question (str): The user's question.

    Returns:
        list: A list of (node_name, node_output) tuples from the graph execution.
    """
    results = []
    try:
        for output in app.stream({"question": question}):
            for node_name, node_output in output.items():
                results.append((node_name, node_output))
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
            results.append(("error", {"generation": "⚠️ **Groq API Rate Limit Exceeded** ⚠️\n\nI have reached the maximum number of daily tokens allowed on the free Groq tier. Please try again tomorrow!"}))
        else:
            results.append(("error", {"generation": f"❌ **An unexpected error occurred:**\n\n```text\n{error_msg}\n```"}))
    return results


@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts. Sends a welcome message."""
    await cl.Message(
        content=(
            "Hi there! 👋 I'm **Mauricio Mendoza's AI**.\n\n"
            "I was built using LangGraph and RAG, and I've read all about Mauricio's background, skills, and projects.\n"
            "What would you like to know about his experience?\n\n"
            "*(I'll search my knowledge base to give you the most accurate answer!)*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages.
    Runs the LangGraph agent in a background thread to avoid blocking,
    then displays intermediate steps and the final answer.
    """
    question = message.content

    # Run the synchronous LangGraph in a thread so it doesn't block Chainlit
    results = await asyncio.to_thread(run_graph, question)

    # Show each node as a "Step" in the UI
    for node_name, node_output in results:
        async with cl.Step(name=node_name, type="tool") as step:
            if node_name == "retrieve":
                docs = node_output.get("documents", [])
                step.output = f"Found {len(docs)} document(s) in the vectorstore."

            elif node_name == "grade_documents":
                docs = node_output.get("documents", [])
                step.output = (
                    f"{len(docs)} document(s) passed relevance check."
                )

            elif node_name == "generate":
                step.output = "Answer generated successfully."

            else:
                step.output = f"Node '{node_name}' completed."

    # Extract the final generation or error message
    generation = ""
    for node_name, node_output in reversed(results):
        if node_name == "generate" or node_name == "error":
            generation = node_output.get("generation", "")
            break

    # Send the final answer
    if generation:
        await cl.Message(content=generation).send()
    else:
        await cl.Message(content="I couldn't generate an answer. Please try rephrasing your question.").send()
