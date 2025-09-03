
# run_dev.py
# !/usr/bin/env python3
"""Development runner script."""

import subprocess
import sys
import os
import threading
import time
from pathlib import Path


def run_backend():
    """Run the FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--reload"
        ], cwd="backend")
    except KeyboardInterrupt:
        print("Backend stopped.")


def run_frontend():
    """Run the Streamlit frontend."""
    print("🎨 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ], cwd="frontend")
    except KeyboardInterrupt:
        print("Frontend stopped.")


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "pandas",
        "datasketch", "plotly", "requests"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print("✅ All required packages are installed")
    return True


def setup_project_structure():
    """Set up the project directory structure."""
    dirs = ["backend", "frontend", "logs", "data"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    # Create __init__.py files
    for dir_name in ["backend", "frontend"]:
        init_file = Path(dir_name) / "__init__.py"
        init_file.touch()


def main():
    """Main development runner."""
    print("🔍 MinHash-LSH Text Clustering - Development Mode")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Setup project structure
    setup_project_structure()

    # Start both services in separate threads
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)

    try:
        backend_thread.start()
        time.sleep(2)  # Give backend time to start
        frontend_thread.start()

        print("\n🎉 Services started!")
        print("📡 Backend API: http://localhost:8002")
        print("🌐 Frontend UI: http://localhost:8501")
        print("📚 API Docs: http://localhost:8002/docs")
        print("\nPress Ctrl+C to stop all services...")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        print("Thank you for using MinHash-LSH Clustering!")


if __name__ == "__main__":
    main()