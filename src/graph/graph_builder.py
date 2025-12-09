import networkx as nx
import spacy
import pickle
import yaml
import community as community_louvain 
from sentence_transformers import SentenceTransformer
import ollama
import os

class KnowledgeGraphBuilder:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.nlp = spacy.load(self.config['models']['spacy_model'])
        self.graph = nx.Graph()
        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])
        
    def load_chunks(self):
        if not os.path.exists("processed/chunks.pkl"):
            raise FileNotFoundError("Chunks file not found. Run chunking first.")
            
        with open("processed/chunks.pkl", "rb") as f:
            return pickle.load(f)

    def extract_entities(self, chunk):
        """Extract Named Entities using SpaCy (Nodes)"""
        doc = self.nlp(chunk)
        # Filtering for specific entity types relevant to the domain
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"]]
        return list(set(entities)) # Deduplicate

    def build_graph(self):
        """Construct graph with Nodes and Edges"""
        chunks = self.load_chunks()
        print("Building Graph Nodes and Edges...")
        
        self.chunk_map = [] 

        for i, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk)
            self.chunk_map.append({"id": i, "text": chunk, "entities": entities, "embedding": None})
            
            # Add nodes
            for entity in entities:
                self.graph.add_node(entity)
            
            # Add edges (connect all entities found in the same chunk)
            for j in range(len(entities)):
                for k in range(j + 1, len(entities)):
                    if self.graph.has_edge(entities[j], entities[k]):
                        self.graph[entities[j]][entities[k]]['weight'] += 1
                    else:
                        self.graph.add_edge(entities[j], entities[k], weight=1)
        
        # Calculate embeddings for all chunks (for retrieval later)
        if self.chunk_map:
            texts = [c["text"] for c in self.chunk_map]
            embeddings = self.encoder.encode(texts)
            for i, emb in enumerate(embeddings):
                self.chunk_map[i]["embedding"] = emb

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def detect_communities(self):
        """Apply community detection (Leiden/Louvain)."""
        print("Detecting Communities...")
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Skipping community detection.")
            self.communities = {}
            return

        # Using Louvain for simplicity and stability
        partition = community_louvain.best_partition(self.graph)
        
        # Group entities by community
        self.communities = {}
        for entity, community_id in partition.items():
            if community_id not in self.communities:
                self.communities[community_id] = []
            self.communities[community_id].append(entity)
            
        print(f"Detected {len(self.communities)} communities.")

    def summarize_communities(self):
        """Generate summaries for each community."""
        print("Summarizing Communities (this may take time)...")
        self.community_summaries = {}
        
        for com_id, entities in self.communities.items():
            # Skip tiny communities to save time
            if len(entities) < 3: 
                continue
                
            # Create a prompt with entities
            entity_list = ", ".join(entities[:20]) # Limit to top 20 to fit context
            prompt = f"Summarize the relationship between these entities in the context of Dr. B.R. Ambedkar's work: {entity_list}"
            
            try:
                # Call Ollama
                response = ollama.chat(model=self.config['models']['llm_model'], messages=[
                    {'role': 'user', 'content': prompt},
                ])
                summary = response['message']['content']
                
                # Store summary and its embedding
                self.community_summaries[com_id] = {
                    "summary": summary,
                    "entities": entities,
                    "embedding": self.encoder.encode(summary)
                }
            except Exception as e:
                print(f"Error summarizing community {com_id}: {e}")
        
    def save(self):
        data = {
            "graph": self.graph,
            "chunk_map": self.chunk_map, 
            "communities": self.communities,
            "community_summaries": self.community_summaries
        }
        with open("processed/knowledge_graph.pkl", "wb") as f:
            pickle.dump(data, f)
        print("Knowledge Graph saved.")