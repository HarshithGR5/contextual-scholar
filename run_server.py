#!/usr/bin/env python3
"""
Simple standalone server for testing the UI
"""
import sys
import os
import uvicorn

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_server():
    """Run the server directly."""
    print("🚀 Starting Contextual Scholar Server...")
    print("=" * 50)
    
    try:
        # Import after path setup
        from app.main import app
        
        # Run the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_server())
