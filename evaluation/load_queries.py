import json

def load_queries(path="evaluation/evaluation_queries.json"):
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    return queries