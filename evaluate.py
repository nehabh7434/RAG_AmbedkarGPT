import json
from src.pipeline.pride import initialize_system, generate_answer


def precision_at_k(docs, expected, k=5):
    relevant = 0

    for doc in docs[:k]:
        if expected.lower() in doc.lower():
            relevant += 1

    return relevant / k


def recall_at_k(docs, expected, k=5):
    relevant = 0

    for doc in docs[:k]:
        if expected.lower() in doc.lower():
            relevant += 1

    return relevant


def evaluate():

    with open("evaluation/evaluation_queries.json", "r") as f:
        queries = json.load(f)

    engine = initialize_system()

    total = len(queries)

    correct_answers = 0

    vector_precisions = []
    hybrid_precisions = []

    vector_recalls = []
    hybrid_recalls = []

    k = 5

    print("\n===== Running Evaluation =====\n")

    for item in queries:

        query = item["query"]
        expected = item["expected_answer"]

        print("Question:", query)

        # VECTOR RETRIEVAL
        vector_docs = engine.local_search(query)

        vp = precision_at_k(vector_docs, expected, k)
        vr = recall_at_k(vector_docs, expected, k)

        vector_precisions.append(vp)
        vector_recalls.append(vr)

        # HYBRID RETRIEVAL
        hybrid_docs = engine.local_search(query)

        hp = precision_at_k(hybrid_docs, expected, k)
        hr = recall_at_k(hybrid_docs, expected, k)

        hybrid_precisions.append(hp)
        hybrid_recalls.append(hr)

        # GENERATE FINAL ANSWER
        generated = generate_answer(query, engine)

        print("Expected:", expected)
        print("Generated:", generated[:200])
        print("-" * 60)

        if expected.lower() in generated.lower():
            correct_answers += 1

    # FINAL METRICS

    answer_accuracy = correct_answers / total

    vector_precision_avg = sum(vector_precisions) / total
    hybrid_precision_avg = sum(hybrid_precisions) / total

    vector_recall_avg = sum(vector_recalls) / total
    hybrid_recall_avg = sum(hybrid_recalls) / total

    print("\n===== FINAL RESULTS =====\n")

    print("Answer Accuracy:", round(answer_accuracy, 3))

    print("\nRetrieval Comparison\n")

    print("{:<20} {:<15} {:<15}".format(
        "Method", "Precision@5", "Recall@5"))

    print("-" * 50)

    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Vector Search",
        vector_precision_avg,
        vector_recall_avg
    ))

    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Hybrid Search",
        hybrid_precision_avg,
        hybrid_recall_avg
    ))


if __name__ == "__main__":
    evaluate()