"""
Test configuration and fixtures
"""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_pdf_content():
    """Sample PDF text content for testing."""
    return """
    Title: Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that provides 
    systems the ability to automatically learn and improve from experience 
    without being explicitly programmed.
    
    Key concepts in machine learning include:
    1. Supervised learning
    2. Unsupervised learning
    3. Reinforcement learning
    
    Neural networks are a fundamental component of deep learning, which is 
    a subset of machine learning.
    """


@pytest.fixture
def sample_research_query():
    """Sample research query for testing."""
    return {
        "question": "What are the main types of machine learning?",
        "top_k": 5,
        "include_entities": True
    }
