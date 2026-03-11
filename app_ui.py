"""
Chainlit UI Application.
Provides an interactive web-based chat interface for the LangGraph RAG Agent.
Run with: chainlit run app_ui.py --port 8080
"""

import asyncio
import chainlit as cl
from src.graph.builder import app
from src.retrieval.document_indexer import format_docs


def run_graph(question: str) -> list:
    """
    Runs the LangGraph agent synchronously and collects all node outputs.

    Args:
        question (str): The user's question.

    Returns:
        list: A list of (node_name, node_output) tuples from the graph execution.
    """
    results = []
    for output in app.stream({"question": question}):
        for node_name, node_output in output.items():
            results.append((node_name, node_output))
    return results


@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts. Sends a welcome message."""
    await cl.Message(
        content=(
            "Welcome to my **Personal AI Assistant**!\n\n"
            "Ask me anything about:\n"
            "- My background and experience\n"
            "- My projects and skills\n"
            "- Anything else you'd like to know about me!\n\n"
            "I'll search my knowledge base to find the best answer."
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

    # Extract the final generation
    generation = ""
    for node_name, node_output in reversed(results):
        if node_name == "generate":
            generation = node_output.get("generation", "")
            break

    # Send the final answer
    if generation:
        await cl.Message(content=generation).send()
    else:
        await cl.Message(content="I couldn't generate an answer. Please try rephrasing your question.").send()
