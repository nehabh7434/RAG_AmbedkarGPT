import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunking.semantic_chunker import SemanticChunker

@pytest.fixture
def chunker():
    """Fixture to initialize chunker with config"""
    return SemanticChunker("config.yaml")

def test_initialization(chunker):
    assert chunker.buffer_size == 1
    assert chunker.max_tokens == 1024

def test_buffer_merge_logic(chunker):
    """Test if buffer merging combines neighbors correctly"""
    sentences = ["A", "B", "C", "D"]
    buffered = chunker._buffer_merge(sentences)
    
    assert "A B" in buffered[0]
    assert "A B C" in buffered[1]
    assert len(buffered) == 4

def test_overlap_splitting(chunker):
    """Test if large text is split with overlap"""
    long_text = "word " * 1200 
    
    # FIX: Update overlap to be smaller than max_tokens
    chunker.max_tokens = 50
    # We must manually update the config dictionary or the method won't know
    chunker.config['chunking']['overlap_tokens'] = 10 
    
    sub_chunks = chunker._split_chunks_with_overlap(long_text)
    
    assert len(sub_chunks) > 1
    assert all(len(c) > 0 for c in sub_chunks)