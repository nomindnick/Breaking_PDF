#!/usr/bin/env python3
"""
Test script for Sprint 1 API implementation.

Tests basic API functionality including health checks and file upload.
"""
import asyncio
import sys
from pathlib import Path

import requests

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test basic health check endpoint."""
    print("Testing basic health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["success"] is True
        print("✅ Basic health check passed\n")
    except Exception as e:
        print(f"❌ Basic health check failed: {e}\n")
        return False
    return True


def test_detailed_health_check():
    """Test detailed health check endpoint."""
    print("Testing detailed health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health/detailed")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"API Version: {data['api']['version']}")
        print(f"System CPU: {data['system']['cpu']['percent']}%")
        print(f"System Memory: {data['system']['memory']['percent']}%")
        assert response.status_code == 200
        assert "status" in data
        print("✅ Detailed health check passed\n")
    except Exception as e:
        print(f"❌ Detailed health check failed: {e}\n")
        return False
    return True


def test_file_upload():
    """Test file upload endpoint."""
    print("Testing file upload...")

    # Find a test PDF file
    test_files = [
        Path("Test_PDF_Set_1.pdf"),
        Path("Test_PDF_Set_2_ocr.pdf"),
        Path("comprehensive_test_pdf.pdf"),
    ]

    test_file = None
    for f in test_files:
        if f.exists():
            test_file = f
            break

    if not test_file:
        print("❌ No test PDF file found")
        return False

    print(f"Using test file: {test_file}")

    try:
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/api/upload", files=files)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print(f"Upload ID: {data['upload_id']}")
            print(f"File size: {data['file_size']} bytes")
            print(f"Total pages: {data['total_pages']}")
            print("✅ File upload passed\n")
            return data["upload_id"]
        else:
            print(f"❌ File upload failed: {response.text}\n")
            return None
    except Exception as e:
        print(f"❌ File upload failed with exception: {e}\n")
        return None


def test_upload_status(upload_id):
    """Test upload status endpoint."""
    print(f"Testing upload status for ID: {upload_id}...")
    try:
        response = requests.get(f"{BASE_URL}/api/upload/{upload_id}/status")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Upload status: {data['status']}")
            print(f"File name: {data['file_name']}")
            print(f"Created at: {data['created_at']}")
            print("✅ Upload status check passed\n")
            return True
        else:
            print(f"❌ Upload status check failed: {response.text}\n")
            return False
    except Exception as e:
        print(f"❌ Upload status check failed with exception: {e}\n")
        return False


def test_invalid_file_upload():
    """Test uploading an invalid file type."""
    print("Testing invalid file upload...")
    try:
        # Create a temporary text file
        temp_file = Path("test.txt")
        temp_file.write_text("This is not a PDF")

        with open(temp_file, "rb") as f:
            files = {"file": (temp_file.name, f, "text/plain")}
            response = requests.post(f"{BASE_URL}/api/upload", files=files)

        temp_file.unlink()  # Clean up

        print(f"Status: {response.status_code}")
        if response.status_code == 415:  # Unsupported Media Type
            print("✅ Invalid file type correctly rejected\n")
            return True
        else:
            print(f"❌ Invalid file type not rejected properly: {response.text}\n")
            return False
    except Exception as e:
        print(f"❌ Invalid file upload test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Sprint 1 API Tests")
    print("=" * 60)
    print()

    # Check if API is running
    try:
        response = requests.get(BASE_URL)
        print(f"API is running at {BASE_URL}")
        print(f"API info: {response.json()}")
        print()
    except requests.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("Please start the API with: python -m pdf_splitter.api.main")
        return 1

    # Run tests
    all_passed = True

    if not test_health_check():
        all_passed = False

    if not test_detailed_health_check():
        all_passed = False

    upload_id = test_file_upload()
    if upload_id:
        if not test_upload_status(upload_id):
            all_passed = False
    else:
        all_passed = False

    if not test_invalid_file_upload():
        all_passed = False

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
