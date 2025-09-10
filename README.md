# Contextual Scholar: AI-Powered Research Assistant

A sophisticated RAG (Retrieval-Augmented Generation) application that combines vector similarity search with knowledge graphs to provide context-aware research assistance.

## 🚀 Features

- **Semantic Document Retrieval**: SentenceTransformers embeddings with ChromaDB vector database
- **Knowledge Graph Integration**: Neo4j graph database for entity relationships and enhanced context
- **AI-Powered Generation**: Google Gemini 2.0 Flash API for intelligent response generation
- **Web Interface**: Modern React-style UI for document upload and querying
- **Multi-format Support**: PDF processing and document chunking
- **Citation Tracking**: Automatic source attribution and reference management

## 🔧 Tech Stack

- **Backend**: FastAPI with Python 3.8+
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB for semantic search
- **Knowledge Graph**: Neo4j for entity relationships
- **LLM**: Google Gemini 2.0 Flash API
- **Frontend**: HTML5 + Modern CSS + JavaScript
- **Processing**: PyPDF2 for document ingestion

## 📁 Project Structure

```
contextual-scholar/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic schemas
│   ├── services/            # Core business logic
│   │   ├── embeddings.py    # Embedding service
│   │   ├── vector_store.py  # Vector operations
│   │   ├── knowledge_graph.py # Graph operations
│   │   ├── llm_service.py   # LLM integration
│   │   └── rag_pipeline.py  # RAG orchestration
│   ├── api/                 # API endpoints
│   ├── templates/           # Web UI templates
│   ├── static/              # CSS/JS assets
│   └── utils/               # Utilities
├── data/                    # Document storage
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
└── docker-compose.yml      # Container setup
```

## 🛠️ Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd contextual-scholar
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# GEMINI_API_KEY=your_gemini_api_key
# NEO4J_URI=neo4j://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password
```

### 4. Start Services

**Option A: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option B: Manual**
```bash
# Start Neo4j (install Neo4j Desktop first)
# Start the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Access Application
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔑 API Endpoints

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### Query Documents
```bash
POST /query
Content-Type: application/json

{
  "query": "What are the main findings about climate change?",
  "top_k": 5
}
```

### Health Check
```bash
GET /health
```

## 🎯 Usage Examples

### Basic Query
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "Explain the methodology used in the research",
    "top_k": 5
})

print(response.json()["answer"])
```

### Upload and Query
```python
# Upload document
files = {"file": open("research.pdf", "rb")}
upload_response = requests.post("http://localhost:8000/upload", files=files)

# Query the document
query_response = requests.post("http://localhost:8000/query", json={
    "query": "What are the key conclusions?"
})
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB storage path

### Model Configuration
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Top-K Results**: 5 (configurable)

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run
docker-compose up --build -d

# Scale services
docker-compose up --scale app=3
```

### Manual Deployment
```bash
# Production server
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## � Monitoring

- **Health Endpoint**: `/health` - Service status
- **Metrics**: Built-in FastAPI metrics
- **Logs**: Structured logging with timestamps

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector database capabilities
- **SentenceTransformers** for embedding models
- **Neo4j** for graph database functionality
- **Google Gemini** for AI generation
- **FastAPI** for web framework

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 2. Database Setup

**Neo4j Setup:**
```bash
# Using Docker (recommended)
docker run -d --name neo4j-contextual-scholar \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Access Neo4j browser at http://localhost:7474
```

**ChromaDB:**
ChromaDB will be automatically initialized when the application starts.

### 3. Configuration

Create a `.env` file:
```env
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Application Configuration
DEBUG=True
```

### 4. Run the Application

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔄 Usage

### Data Ingestion

1. Place PDF documents in the `data/` directory
2. Use the ingestion endpoint to process documents:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/research_paper.pdf"}'
```

### Query the Assistant

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Retrieval-Augmented Generation?",
    "top_k": 5
  }'
```

Response format:
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is...",
  "sources": [
    {
      "doc_id": "paper_123",
      "title": "RAG: Retrieval-Augmented Generation",
      "score": 0.95,
      "chunk": "..."
    }
  ],
  "related_entities": [
    {
      "entity": "Neural Information Retrieval",
      "relationship": "RELATED_TO",
      "context": "..."
    }
  ]
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 🌟 Advanced Features

### Multi-hop Reasoning
The system can handle complex queries that require traversing the knowledge graph:
```
"Which researchers collaborated on RAG papers published in 2021?"
```

### Entity-Enhanced Context
Automatically enriches responses with related entities from the knowledge graph for comprehensive answers.

### Configurable Retrieval
Adjust semantic search parameters and graph traversal depth for optimal results.

## 📊 API Documentation

Once running, visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🚀 Deployment

### Docker Deployment
```bash
# Build the image
docker build -t contextual-scholar .

# Run with docker-compose
docker-compose up -d
```

### Production Considerations
- Use environment-specific configuration
- Set up proper logging and monitoring
- Configure database persistence
- Implement rate limiting and authentication

## 📈 Performance Optimization

- **Embedding Caching**: Cached embeddings for faster retrieval
- **Batch Processing**: Efficient document ingestion
- **Index Optimization**: Optimized vector and graph indexes
- **Connection Pooling**: Database connection management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
