# File: src/retrieval/retrieval_engine.py
import pickle
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi # New Library

class RetrievalEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load KG
        with open("processed/knowledge_graph.pkl", "rb") as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            self.chunk_map = data["chunk_map"]
            self.community_summaries = data["community_summaries"]
            
        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])
        
        # --- NEW: Initialize BM25 for Hybrid Search ---
        print("Initializing Hybrid Search (BM25)...")
        corpus = [chunk['text'] for chunk in self.chunk_map]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # ----------------------------------------------

    def _cosine_sim(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    def local_search(self, query):
        """
        Hybrid Search: Combines Vector Similarity (Cosine) with Keyword Matching (BM25)
        """
        query_embedding = self.encoder.encode(query)
        top_k = self.config['retrieval']['top_k_local']
        
        # 1. Vector Scores
        vector_scores = []
        for i, chunk in enumerate(self.chunk_map):
            sim = self._cosine_sim(query_embedding, chunk['embedding'])
            vector_scores.append(sim)
            
        # 2. BM25 Scores (Keyword Match)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1 range to match Cosine
        if max(bm25_scores) > 0:
            bm25_scores = [s / max(bm25_scores) for s in bm25_scores]
            
        # 3. Hybrid Combination (0.7 Vector + 0.3 Keyword)
        hybrid_scores = []
        for i in range(len(self.chunk_map)):
            final_score = (0.7 * vector_scores[i]) + (0.3 * bm25_scores[i])
            hybrid_scores.append((final_score, self.chunk_map[i]['text']))
            
        # Sort and return Top-K
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in hybrid_scores[:top_k]]

    def global_search(self, query):
        # (Same as before - Equation 5)
        query_embedding = self.encoder.encode(query)
        top_k = self.config['retrieval']['top_k_global']
        
        scored_communities = []
        for com_id, data in self.community_summaries.items():
            summary_embedding = data['embedding']
            sim = self._cosine_sim(query_embedding, summary_embedding)
            scored_communities.append((sim, data['summary']))
            
        scored_communities.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in scored_communities[:top_k]]