# MinHash-LSH Text Clustering Application (Python)

A comprehensive Python application for clustering similar texts using the MinHash-LSH (Locality Sensitive Hashing) algorithm with configurable Jaccard similarity thresholds.

## 🌟 Features

- **🔍 Advanced MinHash-LSH Implementation**: Efficient similarity detection for large text datasets
- **⚙️ Configurable Jaccard Threshold**: Adjust similarity sensitivity (0.1-0.9)
- **🎨 Beautiful Streamlit Frontend**: Interactive web interface with real-time visualizations
- **⚡ FastAPI Backend**: High-performance REST API with automatic documentation
- **📊 Interactive Analytics**: Cluster distribution plots and certainty analysis
- **📥 Multiple Export Formats**: Download results as CSV or JSON
- **🐳 Docker Ready**: Complete containerization with Docker Compose
- **🧪 Comprehensive Testing**: Unit tests and performance benchmarks
- **📈 Quality Metrics**: Clustering evaluation with ARI and NMI scores

## 🏗️ Architecture

```
minhash-clustering-python/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Backend dependencies
│   └── Dockerfile.backend      # Backend container
├── frontend/
│   ├── streamlit_app.py        # Streamlit application
│   ├── streamlit_requirements.txt  # Frontend dependencies
│   └── Dockerfile.frontend     # Frontend container
├── tests/
│   ├── test_api.py            # API tests
│   └── benchmark.py           # Performance benchmarks
├── scripts/
│   └── run_dev.py             # Development runner
├── docker-compose.yml         # Production deployment
├── docker-compose.dev.yml     # Development deployment
├── setup.py                   # Package configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone and setup**:
```bash
git clone <repository>
cd minhash-clustering-python
```

2. **Create project structure**:
```bash
mkdir -p backend frontend tests scripts
```

3. **Copy files to respective directories**:
```bash
# Backend files
cp main.py requirements.txt Dockerfile.backend backend/

# Frontend files  
cp streamlit_app.py streamlit_requirements.txt Dockerfile.frontend frontend/

# Test files
cp test_api.py benchmark.py tests/

# Scripts
cp run_dev.py scripts/
```

4. **Deploy with Docker**:
```bash
docker-compose up --build
```

5. **Access the application**:
- 🌐 **Frontend**: http://localhost:8501
- 📡 **Backend API**: http://localhost:8002
- 📚 **API Documentation**: http://localhost:8002/docs

### Option 2: Local Development

1. **Install Python dependencies**:
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend (in a new terminal)
cd frontend
pip install -r streamlit_requirements.txt
```

2. **Run services**:
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# Terminal 2: Frontend
cd frontend
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

3. **Or use the development runner**:
```bash
python scripts/run_dev.py
```

## 📖 Usage Guide

### 1. Prepare Your Data

Create a CSV file with at least one column named `text`:

```csv
text,category,id
"Machine learning algorithms are powerful tools for data analysis",tech,1
"Artificial intelligence and ML techniques revolutionize analytics",tech,2
"The weather today is sunny and warm with clear skies",weather,3
"Beautiful sunny day with pleasant temperatures expected",weather,4
"Python programming language is great for data science",programming,5
"Java is another popular programming language for development",programming,6
```

### 2. Upload and Configure

1. **Upload CSV**: Use the file uploader in the sidebar
2. **Set Jaccard Threshold**: Adjust the slider (0.1-0.9)
   - **Lower values (0.1-0.3)**: More permissive clustering, larger clusters
   - **Higher values (0.7-0.9)**: Stricter similarity, smaller clusters
3. **Preview Data**: Expand the CSV preview to verify your data

### 3. Cluster and Analyze

1. **Click "Cluster Texts"**: Processing time varies with dataset size
2. **View Results**: Interactive table with sortable columns
3. **Analyze Clusters**: Distribution plots and certainty metrics
4. **Export Data**: Download as CSV or JSON

### 4. Interpret Results

**Certainty Levels:**
- 🟢 **High (≥80%)**: Strong confidence in cluster assignment
- 🟡 **Medium (60-79%)**: Moderate confidence 
- 🔴 **Low (<60%)**: Weak confidence or singleton clusters

## 🔧 Algorithm Details

### MinHash-LSH Implementation

1. **Text Preprocessing**: 
   - Convert to lowercase
   - Remove punctuation
   - Generate 3-gram shingles

2. **MinHash Signatures**:
   - 128 hash functions (configurable)
   - Compact document representation
   - Preserves Jaccard similarity

3. **LSH Bucketing**:
   - Efficient candidate pair generation
   - Configurable threshold parameter
   - Reduces comparison complexity from O(n²) to O(n×c)

4. **Exact Similarity Calculation**:
   - Jaccard similarity for candidate pairs
   - User-configurable threshold filtering

5. **Union-Find Clustering**:
   - Connected components algorithm
   - Optimal cluster formation
   - Linear time complexity

6. **Certainty Scoring**:
   - Average similarity to cluster members
   - Confidence measure for assignments

### Performance Characteristics

- **Time Complexity**: O(n×c) average case, where c is candidate pairs
- **Space Complexity**: O(n×h) for n documents and h hash functions
- **Scalability**: Optimized for datasets up to 100K documents
- **Memory Efficient**: MinHash signatures reduce memory footprint

## 📊 API Reference

### FastAPI Endpoints

#### `POST /api/cluster`
Cluster texts from uploaded CSV file.

**Parameters:**
- `file`: CSV file (multipart/form-data)
- `jaccard_threshold`: Float (0.0-1.0)

**Response:**
```json
[
  {
    "text": "Sample document text",
    "cluster_id": 0,
    "certainty": 0.85,
    "original_index": 0
  }
]
```

#### `POST /api/download`
Download clustering results as CSV.

**Body:** JSON array of clustered documents
**Response:** CSV file stream

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "MinHash LSH Clustering"
}
```

### Error Handling

- **400 Bad Request**: Invalid file format, missing text column, invalid threshold
- **500 Internal Server Error**: Processing errors, memory issues
- **Detailed Error Messages**: Comprehensive error descriptions for debugging

## 🧪 Testing and Benchmarking

### Run Tests

```bash
# Unit tests
cd tests
pytest test_api.py -v

# Performance benchmark
python benchmark.py
```

### Benchmark Results

The application includes comprehensive benchmarking tools:

- **Performance Analysis**: Processing time vs dataset size
- **Quality Metrics**: ARI and NMI scores against ground truth
- **Memory Usage**: Resource consumption analysis
- **Scalability Tests**: Performance with different parameters

Expected performance (on modern hardware):
- **1K documents**: ~2-5 seconds
- **10K documents**: ~30-60 seconds  
- **50K documents**: ~5-10 minutes

## ⚙️ Configuration

### MinHash-LSH Parameters

Adjust in `main.py`:

```python
# For larger datasets
MinHashLSHClustering(num_perm=256, threshold=0.3)

# For faster processing
MinHashLSHClustering(num_perm=64, threshold=0.3)

# Modify shingle size in generate_shingles()
def generate_shingles(self, text: str, k: int = 5):  # 5-grams for longer documents
```

### Streamlit Configuration

Modify `streamlit_app.py`:

```python
# Page configuration
st.set_page_config(
    page_title="Custom Title",
    page_icon="🔍",
    layout="wide"
)

# Threshold range
jaccard_threshold = st.slider(
    "Jaccard Similarity Threshold",
    min_value=0.05,  # Lower minimum
    max_value=0.95,  # Higher maximum
    value=0.3,
    step=0.05
)
```

### Docker Configuration

Environment variables:
```bash
# Backend
export PYTHONUNBUFFERED=1
export WORKERS=4
export PORT=8002

# Frontend  
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🚨 Troubleshooting

### Common Issues

1. **"No 'text' column found"**
   - Ensure CSV has a column named 'text' (case-insensitive)
   - Check for extra spaces in column names

2. **Connection errors**
   - Verify backend is running on port 8002
   - Check firewall settings
   - Ensure Docker containers are networked properly

3. **Memory errors**
   - Reduce dataset size
   - Increase Docker memory limits
   - Lower num_perm parameter

4. **Slow processing**
   - Reduce dataset size for testing
   - Increase Jaccard threshold
   - Use fewer hash functions

5. **Empty clusters**
   - Lower Jaccard threshold
   - Check text quality and diversity
   - Verify sufficient text length

### Docker Issues

```bash
# View logs
docker-compose logs backend
docker-compose logs frontend

# Rebuild containers
docker-compose down
docker-compose up --build

# Check container status
docker-compose ps
```

### Development Issues

```bash
# Install missing packages
pip install -r requirements.txt
pip install -r streamlit_requirements.txt

# Check Python version (requires 3.8+)
python --version

# Run in debug mode
uvicorn main:app --reload --log-level debug
```

## 🎯 Production Deployment

### Security Considerations

1. **Add Authentication**:
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()
```

2. **Input Validation**:
```python
from pydantic import BaseModel, validator
```

3. **Rate Limiting**:
```python
from slowapi import Limiter
```

4. **HTTPS Configuration**:
```bash
# Use reverse proxy (nginx/traefik)
# Add SSL certificates
```

### Performance Optimization

1. **Async Processing**:
```python
import asyncio
from fastapi import BackgroundTasks
```

2. **Caching**:
```python
from functools import lru_cache
```

3. **Database Integration**:
```python
from sqlalchemy import create_engine
```

4. **Message Queue**:
```python
from celery import Celery
```

### Monitoring

1. **Add Metrics**:
```python
from prometheus_client import Counter, Histogram
```

2. **Health Checks**:
```python
# Extended health check with dependencies
```

3. **Logging**:
```python
import structlog
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Run the test suite
5. Submit a pull request

### Code Style

```bash
# Format code
black .
isort .

# Lint code  
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **DataSketch**: Excellent MinHash-LSH implementation
- **FastAPI**: Modern, fast web framework
- **Streamlit**: Beautiful data apps framework
- **Plotly**: Interactive visualization library

## 📞 Support

- 📖 **Documentation**: Check this README and inline comments
- 🐛 **Issues**: Create GitHub issues for bugs
- 💬 **Discussions**: Use GitHub discussions for questions
- 📧 **Contact**: your.email@example.com

---

Built with ❤️ using Python, FastAPI, and Streamlit | MinHash-LSH Algorithm Implementation