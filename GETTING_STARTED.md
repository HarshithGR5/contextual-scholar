# Getting Started with Contextual Scholar

This guide will help you set up and run the Contextual Scholar RAG application.

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional, for containerized setup)
- Neo4j database (or use Docker setup)

### 2. Installation

#### Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd contextual-scholar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

#### Docker Setup

```bash
# Clone the repository
git clone <repository-url>
cd contextual-scholar

# Set up environment variables
cp .env.example .env
# Edit .env with your Gemini API key

# Start all services
docker-compose up -d
```

### 3. Configuration

Edit the `.env` file with your configuration:

- **GEMINI_API_KEY**: Your Google Gemini API key
- **NEO4J_URI**: Neo4j connection URI
- **NEO4J_USER/NEO4J_PASSWORD**: Neo4j credentials

### 4. Start the Application

#### Local Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health
- **Neo4j Browser** (if using Docker): http://localhost:7474

## API Usage Examples

### 1. Ingest a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/sample_research_paper.pdf",
    "metadata": {"source": "academic", "year": 2024}
  }'
```

### 2. Upload and Ingest a File

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@path/to/your/document.pdf" \
  -F "metadata={\"source\": \"upload\"}"
```

### 3. Ask a Research Question

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Retrieval-Augmented Generation?",
    "top_k": 5,
    "include_entities": true
  }'
```

### 4. Check System Status

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### 5. Get System Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/stats"
```

## Adding Your Own Documents

1. **PDF Files**: Place PDF files in the `data/` directory
2. **Use the Ingest API**: Call the `/api/v1/ingest` endpoint
3. **Upload Interface**: Use the `/api/v1/ingest/upload` endpoint for file uploads

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   FastAPI       │───▶│  RAG Pipeline   │
└─────────────────┘    │   Router        │    └─────────────────┘
                       └─────────────────┘             │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Gemini LLM     │◀───│  Vector Store   │◀───│   Embeddings    │
│  (Generation)   │    │   (ChromaDB)    │    │(SentenceTransf.)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │
         │              ┌─────────────────┐
         └──────────────│ Knowledge Graph │
                        │    (Neo4j)      │
                        └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **ImportError for dependencies**
   - Ensure all packages are installed: `pip install -r requirements.txt`

2. **Neo4j connection failed**
   - Check if Neo4j is running
   - Verify connection credentials in `.env`
   - For graceful degradation, the system will work without Neo4j

3. **Gemini API errors**
   - Verify your API key in `.env`
   - Check API quota and rate limits

4. **Empty responses**
   - Ensure documents are properly ingested
   - Check vector store status via `/api/v1/stats`

### Logs and Debugging

- Set `DEBUG=True` in `.env` for detailed logging
- Check Docker logs: `docker-compose logs contextual-scholar`
- Use the health endpoint to verify service status

## Performance Tips

1. **Batch Processing**: Ingest multiple documents before querying
2. **Optimal Chunk Size**: Adjust `CHUNK_SIZE` in `.env` based on your content
3. **Database Optimization**: Use appropriate indexes for Neo4j queries
4. **Caching**: ChromaDB automatically handles embedding caching

## Security Considerations

- Keep your Gemini API key secure
- Use proper authentication in production
- Configure CORS settings appropriately
- Use HTTPS in production deployments

For more detailed information, see the full README.md file.
