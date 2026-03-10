"""
System Entrypoint.
Orchestrates the execution of the LangGraph RAG Agent from the command line.
"""

import sys
from src.graph.builder import app

def main():
    """Main execution block. Captures arguments and streams the agent's workflow."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is agent memory?"

    print()
    print("=" * 60)
    print(f"  LANGGRAPH RAG AGENT")
    print(f"  Question: {question}")
    print("=" * 60)
    
    inputs = {"question": question}
    
    # Run the graph using the iterator form `app.stream`
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"  [+] Node completed: {key}")
            
    # Print the final result
    print()
    print("-" * 60)
    print("  FINAL ANSWER")
    print("-" * 60)
    
    final_output = value.get("generation") if "generation" in value else "No generation produced."
    print()
    print(f"  {final_output}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
