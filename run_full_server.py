#!/usr/bin/env python3
"""
Robust FastAPI server launcher for full Contextual Scholar functionality
"""
import sys
import os
import uvicorn
import signal
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class RobustServer:
    """Robust server that handles interruptions gracefully."""
    
    def __init__(self):
        self.should_restart = True
        self.server = None
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.should_restart = False
        if self.server:
            self.server.should_exit = True
    
    def run(self):
        """Run the server with restart capability."""
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 Starting Full Contextual Scholar Server...")
        print("=" * 60)
        print("🧠 Mode: Full AI-Powered Research Assistant")
        print("📊 Features: RAG Pipeline + Knowledge Graph + Vector Search")
        print("🔗 URL: http://127.0.0.1:8000")
        print("📖 API Docs: http://127.0.0.1:8000/docs")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        retry_count = 0
        max_retries = 3
        
        while self.should_restart and retry_count < max_retries:
            try:
                # Import the FastAPI app
                from app.main import app
                
                # Configure uvicorn
                config = uvicorn.Config(
                    app=app,
                    host="127.0.0.1",
                    port=8000,
                    log_level="info",
                    access_log=True,
                    reload=False,  # Disable reload to prevent interruptions
                    workers=1
                )
                
                self.server = uvicorn.Server(config)
                
                print(f"🔄 Starting server (attempt {retry_count + 1})...")
                
                # Run the server
                self.server.run()
                
                # If we get here, server stopped normally
                break
                
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
                break
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Server error (attempt {retry_count}): {e}")
                
                if retry_count < max_retries:
                    print(f"🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("❌ Max retries reached. Please check the error above.")
                    return False
        
        print("\n✅ Server shutdown complete")
        return True

def main():
    """Main entry point."""
    server = RobustServer()
    
    try:
        success = server.run()
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
