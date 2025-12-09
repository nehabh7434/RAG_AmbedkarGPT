import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.ambedkargpt1 import initialize_system, generate_answer

def test_full_pipeline_flow():
    """
    Test the end-to-end flow. 
    Note: This might be slow as it loads models.
    """
    # 1. Initialize
    engine = initialize_system()
    assert engine is not None
    
    # 2. Test Generation (Mocking Ollama would be better, but for this assignment live is fine)
    # We use a very simple query to save time
    query = "What is endogamy?"
    
    # Just ensure it doesn't crash
    try:
        # We don't assert the answer content because LLM output varies
        # We just check if it runs without throwing an exception
        # (This assumes Ollama is running)
        response = generate_answer(query, engine)
        assert isinstance(response, str)
        assert len(response) > 0
    except Exception as e:
        # If Ollama is off, we catch it, but fail if it's a code error
        if "Connection refused" in str(e):
            pytest.skip("Ollama server not running, skipping integration test")
        else:
            raise e