import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# --- UPDATED IMPORT HERE ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
# ---------------------------
import spacy
from pypdf import PdfReader
import os

class SemanticChunker:
    def __init__(self, config_path="config.yaml"):
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])
        try:
            self.nlp = spacy.load(self.config['models']['spacy_model'])
        except OSError:
            print(f"Spacy model '{self.config['models']['spacy_model']}' not found. Downloading...")
            from spacy.cli import download
            download(self.config['models']['spacy_model'])
            self.nlp = spacy.load(self.config['models']['spacy_model'])
        
        # Hyperparameters
        self.buffer_size = self.config['chunking']['buffer_size']
        self.threshold = self.config['chunking']['breakpoint_threshold']
        self.max_tokens = self.config['chunking']['max_tokens']
        
    def load_pdf(self):
        """Step 1: Load PDF and split into raw sentences."""
        pdf_path = self.config['paths']['pdf_path']
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
            
        # Basic sentence splitting using Spacy
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        return sentences

    def _buffer_merge(self, sentences):
        """
        Implements 'BufferMerge' from the algorithm.
        Combines a sentence with its neighbors to preserve context for embedding.
        """
        buffered_sentences = []
        for i in range(len(sentences)):
            # Create a window of sentences around i
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            
            # Combine sentences in the buffer window
            combined_text = " ".join(sentences[start:end])
            buffered_sentences.append(combined_text)
            
        return buffered_sentences

    def _split_chunks_with_overlap(self, chunk_text):
        """
        Handles chunks exceeding token limits.
        Splits >1024 tokens into sub-chunks with 128 token overlap.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.config['chunking']['overlap_tokens'],
            length_function=len 
        )
        return splitter.split_text(chunk_text)

    def process(self):
        """Main execution of the Chunking Algorithm"""
        print("Loading and splitting PDF...")
        sentences = self.load_pdf()
        
        print("Applying Buffer Merge...")
        # buffered_sentences are used ONLY for embedding calculation
        buffered_sentences = self._buffer_merge(sentences)
        
        print("Generating Embeddings...")
        embeddings = self.encoder.encode(buffered_sentences)
        
        print("Calculating Cosine Distances and Grouping...")
        chunks = []
        current_chunk = [sentences[0]]
        
        # Iterate through embeddings to find breakpoints
        for i in range(len(embeddings) - 1):
            # Calculate cosine distance: d = 1 - similarity
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            distance = 1 - sim
            
            # Check threshold
            if distance < self.threshold:
                current_chunk.append(sentences[i+1])
            else:
                # Breakpoint found, store current chunk and start new
                full_chunk_text = " ".join(current_chunk)
                
                # Check token limits
                if len(full_chunk_text) > self.max_tokens:
                    sub_chunks = self._split_chunks_with_overlap(full_chunk_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(full_chunk_text)
                
                current_chunk = [sentences[i+1]]
        
        # Add the last chunk
        if current_chunk:
             chunks.append(" ".join(current_chunk))
             
        print(f"Generated {len(chunks)} semantic chunks.")
        return chunks