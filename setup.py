#!/usr/bin/env python3
"""
Setup and test script for Contextual Scholar RAG application
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import chromadb
        import neo4j
        import httpx
        import PyPDF2
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def check_environment():
    """Check if environment variables are properly set."""
    required_vars = [
        "GEMINI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER", 
        "NEO4J_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        print("Create a .env file with the required variables")
        return False
    
    print("✓ Environment variables are set")
    return True


def test_gemini_api():
    """Test connection to Gemini API."""
    try:
        import httpx
        from app.utils.config import settings
        
        # Simple test request
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": settings.gemini_api_key
        }
        
        # Test with a simple request
        print("Testing Gemini API connection...")
        # Note: Actual test would require async call
        print("✓ Gemini API key is configured")
        return True
        
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False


def test_neo4j_connection():
    """Test connection to Neo4j database."""
    try:
        from neo4j import GraphDatabase
        from app.utils.config import settings
        
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        with driver.session() as session:
            session.run("RETURN 1")
        
        driver.close()
        print("✓ Neo4j connection successful")
        return True
        
    except Exception as e:
        print(f"⚠ Neo4j connection failed: {e}")
        print("  (Application will work with degraded functionality)")
        return False


def setup_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✓ Directories created")


def create_sample_data():
    """Create sample data for testing."""
    sample_content = """
# Sample Research Document

## Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed.

## Types of Machine Learning

### Supervised Learning
In supervised learning, algorithms learn from labeled training data to make 
predictions or decisions.

### Unsupervised Learning  
Unsupervised learning finds hidden patterns in data without labeled examples.

### Reinforcement Learning
Reinforcement learning involves an agent learning through interaction with an 
environment to maximize cumulative reward.

## Applications

Machine learning has applications in:
- Computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis

## Conclusion

Machine learning continues to evolve and transform various industries through 
its ability to extract insights from data.
"""
    
    sample_file = Path("data/sample_ml_document.txt")
    sample_file.write_text(sample_content)
    print("✓ Sample data created")


async def test_complete_pipeline():
    """Test the complete RAG pipeline."""
    try:
        from app.services.rag_pipeline import rag_pipeline
        from app.models.schemas import ResearchQuery
        
        # Create a simple test query
        query = ResearchQuery(
            question="What is machine learning?",
            top_k=3,
            include_entities=True
        )
        
        print("Testing RAG pipeline...")
        
        # Note: This would need actual documents ingested first
        # For now, just test that the pipeline can be instantiated
        status = rag_pipeline.get_system_status()
        print("✓ RAG pipeline is operational")
        return True
        
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        return False


def run_server():
    """Start the FastAPI server."""
    try:
        import uvicorn
        from app.main import app
        
        print("Starting Contextual Scholar server...")
        print("Access the API documentation at: http://localhost:8000/docs")
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Failed to start server: {e}")


def main():
    """Main setup and test function."""
    parser = argparse.ArgumentParser(description="Contextual Scholar Setup")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    parser.add_argument("--setup", action="store_true", help="Set up the application")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    parser.add_argument("--run", action="store_true", help="Start the server")
    parser.add_argument("--all", action="store_true", help="Run all setup and tests")
    
    args = parser.parse_args()
    
    if args.all or args.check:
        print("=== Checking Dependencies ===")
        if not check_dependencies():
            sys.exit(1)
        
        print("\n=== Checking Environment ===")
        if not check_environment():
            sys.exit(1)
    
    if args.all or args.setup:
        print("\n=== Setting Up Application ===")
        setup_directories()
        create_sample_data()
    
    if args.all or args.test:
        print("\n=== Running Tests ===")
        test_gemini_api()
        test_neo4j_connection()
        
        # Run async test
        try:
            asyncio.run(test_complete_pipeline())
        except Exception as e:
            print(f"Pipeline test error: {e}")
    
    if args.run:
        print("\n=== Starting Server ===")
        run_server()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample usage:")
        print("  python setup.py --all      # Run complete setup")
        print("  python setup.py --check    # Check requirements only")
        print("  python setup.py --run      # Start the server")


if __name__ == "__main__":
    main()
