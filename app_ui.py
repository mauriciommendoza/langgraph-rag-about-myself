"""
Chainlit UI Application.
Provides an interactive web-based chat interface for the LangGraph RAG Agent.
Run with: chainlit run app_ui.py --port 8080
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
from chainlit.input_widget import Select
from src.graph.builder import app
from src.config.llm import AVAILABLE_MODELS, DEFAULT_MODEL


def run_graph(question: str, model_name: str) -> list:
    """
    Runs the LangGraph agent synchronously and collects all node outputs.

    Args:
        question (str): The user's question.
        model_name (str): The selected model identifier.

    Returns:
        list: A list of (node_name, node_output) tuples from the graph execution.
    """
    results = []
    try:
        for output in app.stream({"question": question, "model_name": model_name}):
            for node_name, node_output in output.items():
                results.append((node_name, node_output))
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
            results.append(("error", {"generation": (
                "⚠️ **Rate Limit Exceeded** ⚠️\n\n"
                "I've hit the maximum number of requests allowed on the free Groq tier. "
                "Please wait a minute and try again, or come back later!"
            )}))
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            results.append(("error", {"generation": (
                "⏳ **Request Timed Out** ⏳\n\n"
                "The request took too long to process. "
                "Please try again with a simpler question."
            )}))
        else:
            results.append(("error", {"generation": (
                "❌ **An unexpected error occurred** ❌\n\n"
                f"```text\n{error_msg}\n```\n\n"
                "Please try again later."
            )}))
    return results


@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts. Shows model selector and welcome message."""
    # Set up chat settings with model selector
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="🤖 AI Model",
                values=list(AVAILABLE_MODELS.values()),
                initial_value=DEFAULT_MODEL,
            )
        ]
    ).send()

    # Store the default model in session
    cl.user_session.set("model_name", DEFAULT_MODEL)

    # Find display name for the default model
    display_name = next(
        (name for name, mid in AVAILABLE_MODELS.items() if mid == DEFAULT_MODEL),
        DEFAULT_MODEL,
    )

    await cl.Message(
        content=(
            "Hi there! 👋 I'm **Mauricio Mendoza's AI**.\n\n"
            "I was built using LangGraph and RAG, and I've read all about Mauricio's background, skills, and projects.\n"
            f"Currently using **{display_name}** model. You can change it via the ⚙️ settings icon.\n\n"
            "What would you like to know about his experience?\n\n"
            "*(I'll search my knowledge base to give you the most accurate answer!)*"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Called when the user changes a setting (e.g. model selector)."""
    model_name = settings.get("model", DEFAULT_MODEL)
    cl.user_session.set("model_name", model_name)

    # Find display name
    display_name = next(
        (name for name, mid in AVAILABLE_MODELS.items() if mid == model_name),
        model_name,
    )

    await cl.Message(
        content=f"✅ Model switched to **{display_name}**."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages.
    Runs the LangGraph agent in a background thread to avoid blocking,
    then displays intermediate steps and the final answer.
    """
    question = message.content
    model_name = cl.user_session.get("model_name", DEFAULT_MODEL)

    # Run the synchronous LangGraph in a thread so it doesn't block Chainlit
    results = await asyncio.to_thread(run_graph, question, model_name)

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
