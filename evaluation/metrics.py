def precision_at_k(retrieved_docs, expected_answer, k=5):
    """
    Precision@k = relevant docs in top-k / k
    """
    relevant = 0

    for doc in retrieved_docs[:k]:
        if expected_answer.lower() in doc.lower():
            relevant += 1

    return relevant / k


def recall_at_k(retrieved_docs, expected_answer, total_relevant=1, k=5):
    """
    Recall@k = relevant docs retrieved / total relevant docs
    """
    relevant = 0

    for doc in retrieved_docs[:k]:
        if expected_answer.lower() in doc.lower():
            relevant += 1

    return relevant / total_relevant