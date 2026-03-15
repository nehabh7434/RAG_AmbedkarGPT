def print_comparison(vector_precision, hybrid_precision,
                     vector_recall, hybrid_recall):

    print("\nRetrieval Comparison Table\n")

    print("{:<20} {:<15} {:<15}".format(
        "Method", "Precision@5", "Recall@5"))

    print("-"*50)

    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Vector Search",
        vector_precision,
        vector_recall
    ))

    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Hybrid Search",
        hybrid_precision,
        hybrid_recall
    ))