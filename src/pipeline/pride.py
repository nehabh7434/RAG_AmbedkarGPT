# File: src/pipeline/pride.py
import sys
import os
import yaml
import ollama
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.chunking.semantic_chunker import SemanticChunker
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.retrieval_engine import RetrievalEngine


def initialize_system():
    logging.info("=== Initializing SemRAG System (Pride and Prejudice Dataset) ===")

    # Check if data exists, if not, process it
    if not os.path.exists("processed/chunks.pkl"):
        logging.info("[1/3] Running Semantic Chunking (Algorithm 1)...")
        chunker = SemanticChunker()
        chunks = chunker.process()

        with open("processed/chunks.pkl", "wb") as f:
            import pickle
            pickle.dump(chunks, f)

    if not os.path.exists("processed/knowledge_graph.pkl"):
        logging.info("[2/3] Building Knowledge Graph & Communities...")
        kg = KnowledgeGraphBuilder()
        kg.build_graph()
        kg.detect_communities()
        kg.summarize_communities()
        kg.save()

    logging.info("[3/3] System Ready. Loading Retrieval Engine...")
    return RetrievalEngine()


def generate_answer(query, engine):

    logging.info(f"Query: {query}")

    # Retrieve
    local_context = engine.local_search(query)
    global_context = engine.global_search(query)

    # Construct context
    all_context = local_context + global_context

    print("\n--- Retrieved Context ---")
    for i, chunk in enumerate(all_context):
        print(f"\n[{i}] {chunk[:500]}")

    context_text = "\n".join([f"[{i}] {c}" for i, c in enumerate(all_context)])

    # Prompt
    prompt = f"""
    ...
    """

    # 3. Prompt
    prompt = f"""
You are a question answering system.

Use ONLY the information provided in the context to answer the question.

Rules:
1. Do not use outside knowledge.
2. Do not guess or invent information.
3. If the context does not contain the answer, say:
   "The context does not contain the answer."
4. Answer in 2–3 sentences only.
5. Cite the context number used for your answer.

Context:
{context_text}

Question: {query}

Answer:
"""

    # 4. Generate
    logging.info("...Generating Answer via Ollama")

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": "Answer strictly using the provided context. Do not hallucinate."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response['message']['content']


if __name__ == "__main__":

    # Ensure processed directory exists
    os.makedirs("processed", exist_ok=True)

    engine = initialize_system()

    print("\n=== Pride and Prejudice RAG System ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Ask a question: ")

        if user_query.lower() == 'exit':
            break

        answer = generate_answer(user_query, engine)

        print("\n" + "=" * 50)
        print("RESPONSE:")
        print(answer)
        print("=" * 50 + "\n")