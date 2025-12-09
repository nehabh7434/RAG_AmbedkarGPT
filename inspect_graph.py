import pickle
import networkx as nx

def inspect():
    print("Loading Knowledge Graph...")
    try:
        with open("processed/knowledge_graph.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: File 'processed/knowledge_graph.pkl' not found.")
        return

    # Extract data
    G = data['graph']
    communities = data['communities']
    summaries = data['community_summaries']

    print("\n" + "="*40)
    print(f"📊 GRAPH STATISTICS")
    print("="*40)
    print(f"Total Nodes (Entities): {G.number_of_nodes()}")
    print(f"Total Edges (Relations): {G.number_of_edges()}")
    print(f"Total Communities:      {len(summaries)}")

    print("\n" + "="*40)
    print(f"🧐 SAMPLE ENTITIES (NODES)")
    print("="*40)
    # Print first 20 nodes
    all_nodes = list(G.nodes())
    for i, node in enumerate(all_nodes[:20]):
        print(f"{i+1}. {node}")

    print("\n" + "="*40)
    print(f"🔗 SAMPLE RELATIONSHIPS (EDGES)")
    print("="*40)
    # Print first 10 edges with weights
    all_edges = list(G.edges(data=True))
    for i, (u, v, d) in enumerate(all_edges[:10]):
        print(f"{i+1}. {u} <---> {v} (Weight: {d.get('weight', 1)})")

    print("\n" + "="*40)
    print(f"🏘️ SAMPLE COMMUNITY SUMMARY")
    print("="*40)
    # Print summary of the first community found
    if summaries:
        first_com_id = list(summaries.keys())[0]
        print(f"Community ID: {first_com_id}")
        print(f"Entities: {summaries[first_com_id]['entities'][:5]}...")
        print(f"Summary: {summaries[first_com_id]['summary'][:200]}...") 
    else:
        print("No community summaries found.")

if __name__ == "__main__":
    inspect()