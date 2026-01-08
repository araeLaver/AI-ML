"""
MLOps Pipeline - Main Entry Point
"""

import uvicorn
from src.serving.api import app

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
