
# test_api.py
import pytest
import httpx
import io
import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_test_csv():
    """Create a test CSV file."""
    data = {
        'text': [
            'Machine learning is a powerful tool for data analysis',
            'Artificial intelligence and ML revolutionize analytics',
            'The weather today is sunny and warm',
            'It\'s a beautiful sunny day with clear skies',
            'Python is a programming language',
            'Java is another programming language'
        ],
        'category': ['tech', 'tech', 'weather', 'weather', 'programming', 'programming']
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "MinHash LSH Text Clustering API" in response.json()["message"]


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_clustering_endpoint():
    """Test the clustering endpoint with valid data."""
    csv_content = create_test_csv()

    files = {"file": ("test.csv", csv_content, "text/csv")}
    data = {"jaccard_threshold": 0.3}

    response = client.post("/api/cluster", files=files, data=data)
    assert response.status_code == 200

    results = response.json()
    assert len(results) == 6  # Should have 6 documents

    # Check result structure
    for result in results:
        assert "text" in result
        assert "cluster_id" in result
        assert "certainty" in result
        assert "original_index" in result
        assert 0 <= result["certainty"] <= 1


def test_clustering_invalid_threshold():
    """Test clustering with invalid threshold."""
    csv_content = create_test_csv()

    files = {"file": ("test.csv", csv_content, "text/csv")}
    data = {"jaccard_threshold": 1.5}  # Invalid threshold

    response = client.post("/api/cluster", files=files, data=data)
    assert response.status_code == 400


def test_clustering_no_text_column():
    """Test clustering with CSV without text column."""
    data = {'description': ['test1', 'test2'], 'value': [1, 2]}
    df = pd.DataFrame(data)
    csv_content = df.to_csv(index=False).encode()

    files = {"file": ("test.csv", csv_content, "text/csv")}
    data = {"jaccard_threshold": 0.3}

    response = client.post("/api/cluster", files=files, data=data)
    assert response.status_code == 400
    assert "must contain a column named 'text'" in response.json()["detail"]


def test_download_endpoint():
    """Test the download endpoint."""
    sample_results = [
        {
            "text": "Sample text 1",
            "cluster_id": 0,
            "certainty": 0.85,
            "original_index": 0
        },
        {
            "text": "Sample text 2",
            "cluster_id": 1,
            "certainty": 0.92,
            "original_index": 1
        }
    ]

    response = client.post("/api/download", json=sample_results)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"


if __name__ == "__main__":
    pytest.main([__file__])
