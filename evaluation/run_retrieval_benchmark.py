import json
from evaluation.load_queries import load_queries
from evaluation.metrics import precision_at_k, recall_at_k
from src.pipeline.pride import initialize_system


def evaluate_retrieval():

    engine = initialize_system()

    queries = load_queries()

    k = 5

    vector_precisions = []
    hybrid_precisions = []

    vector_recalls = []
    hybrid_recalls = []

    print("\nRunning Retrieval Benchmark\n")

    for q in queries:

        query = q["query"]
        expected = q["expected_answer"]

        # vector retrieval
        vector_docs = engine.local_search(query)

        # hybrid retrieval
        hybrid_docs = engine.hybrid_search(query)

        p_vec = precision_at_k(vector_docs, expected, k)
        p_hyb = precision_at_k(hybrid_docs, expected, k)

        r_vec = recall_at_k(vector_docs, expected, 1, k)
        r_hyb = recall_at_k(hybrid_docs, expected, 1, k)

        vector_precisions.append(p_vec)
        hybrid_precisions.append(p_hyb)

        vector_recalls.append(r_vec)
        hybrid_recalls.append(r_hyb)

        print("Query:", query)
        print("Expected:", expected)
        print("Vector Top Result:", vector_docs[0][:120])
        print("Hybrid Top Result:", hybrid_docs[0][:120])
        print("-" * 60)

    print("\n====== Final Results ======\n")

    print("Vector Precision@5:", sum(vector_precisions)/len(vector_precisions))
    print("Hybrid Precision@5:", sum(hybrid_precisions)/len(hybrid_precisions))

    print("Vector Recall@5:", sum(vector_recalls)/len(vector_recalls))
    print("Hybrid Recall@5:", sum(hybrid_recalls)/len(hybrid_recalls))


if __name__ == "__main__":
    evaluate_retrieval()