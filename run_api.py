#!/usr/bin/env python3
"""
Simple script to run the PDF Splitter API.
"""
import sys

import uvicorn

from pdf_splitter.api.config import config

if __name__ == "__main__":
    print(f"Starting {config.api_title} v{config.api_version}")
    print(f"API will be available at http://{config.api_host}:{config.api_port}")
    print(f"Documentation at http://{config.api_host}:{config.api_port}/api/docs")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(
        "pdf_splitter.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.reload,
        log_level=config.log_level.lower(),
    )
