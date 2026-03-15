from evaluation.load_queries import load_queries
from src.pipeline.pride import initialize_system, generate_answer


def evaluate_answers():

    engine = initialize_system()

    queries = load_queries()

    correct = 0
    total = len(queries)

    for q in queries:

        query = q["query"]
        expected = q["expected_answer"]

        answer = generate_answer(query, engine)

        print("\nQuestion:", query)
        print("Expected:", expected)
        print("Generated:", answer)

        if expected.lower() in answer.lower():
            correct += 1

    accuracy = correct / total

    print("\nAnswer Accuracy:", accuracy)


if __name__ == "__main__":
    evaluate_answers()