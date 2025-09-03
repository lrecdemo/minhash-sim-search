from setuptools import setup, find_packages

setup(
    name="minhash-lsh-clustering",
    version="1.0.0",
    description="MinHash-LSH Text Clustering Application",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pandas>=2.1.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.2",
        "datasketch>=1.6.4",
        "python-multipart>=0.0.6",
        "streamlit>=1.28.1",
        "requests>=2.31.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
