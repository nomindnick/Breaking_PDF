#!/usr/bin/env python3
"""
Test script for Sprint 5 API implementation.

Tests results viewing, file downloads, and analytics functionality.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# API base URL
BASE_URL = "http://localhost:8000"


def setup_test_session() -> Optional[str]:
    """Set up a test session with completed processing."""
    print("Setting up test session...")

    # Find test PDF
    test_file = None
    for f in [Path("Test_PDF_Set_1.pdf"), Path("Test_PDF_Set_2_ocr.pdf")]:
        if f.exists():
            test_file = f
            break

    if not test_file:
        print("❌ No test PDF file found")
        return None

    # Upload file
    print(f"Uploading {test_file.name}...")
    with open(test_file, "rb") as f:
        files = {"file": (test_file.name, f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return None

    file_id = response.json()["upload_id"]

    # Start processing
    print("Starting processing...")
    response = requests.post(f"{BASE_URL}/api/process", json={"file_id": file_id})

    if response.status_code != 200:
        print(f"❌ Processing failed: {response.text}")
        return None

    session_id = response.json()["session_id"]

    # Wait for processing to complete
    print("Waiting for processing to complete...")
    for i in range(30):
        response = requests.get(f"{BASE_URL}/api/process/{session_id}/status")
        if response.status_code == 200:
            status = response.json()["status"]
            if status == "confirmed":
                # Execute split
                print("Executing split...")
                response = requests.post(f"{BASE_URL}/api/splits/{session_id}/execute")
                if response.status_code == 200:
                    # Wait for split to complete
                    time.sleep(5)
                    return session_id
        time.sleep(1)

    return session_id


def test_get_results(session_id: str):
    """Test getting session results."""
    print(f"\nTesting get results for session {session_id}...")

    response = requests.get(f"{BASE_URL}/api/results/{session_id}")

    if response.status_code == 200:
        results = response.json()
        print("✅ Results retrieved:")
        print(f"  Status: {results['status']}")
        print(f"  Files created: {results['files_created']}")
        print(f"  Total output size: {results['total_output_size']} bytes")
        print(f"  Processing time: {results['processing_time']:.2f}s")

        if results.get("output_files"):
            print(f"  Output files:")
            for file_info in results["output_files"][:3]:
                print(f"    - {file_info['filename']} ({file_info['size']} bytes)")

        return True, results.get("output_files", [])
    else:
        print(f"❌ Get results failed: {response.text}")
        return False, []


def test_search_results():
    """Test searching results."""
    print("\nTesting search results...")

    # Search for recent results
    filter_criteria = {
        "created_after": (datetime.utcnow() - timedelta(days=7)).isoformat()
    }

    response = requests.post(
        f"{BASE_URL}/api/results/search",
        json=filter_criteria,
        params={"page": 1, "page_size": 10},
    )

    if response.status_code == 200:
        page = response.json()
        print("✅ Search results:")
        print(f"  Total results: {page['total']}")
        print(f"  Page {page['page']} of {page['total_pages']}")
        print(f"  Results on this page: {len(page['results'])}")

        return True
    else:
        print(f"❌ Search failed: {response.text}")
        return False


def test_file_preview(session_id: str, filename: str):
    """Test file preview."""
    print(f"\nTesting file preview for {filename}...")

    response = requests.get(
        f"{BASE_URL}/api/results/{session_id}/preview/{filename}",
        params={"preview_type": "text", "max_pages": 2},
    )

    if response.status_code == 200:
        preview = response.json()
        print("✅ Preview generated:")
        print(f"  Preview type: {preview['preview_type']}")
        print(f"  File type: {preview['file_type']}")

        if preview.get("content"):
            print(f"  Content length: {len(preview['content'])} chars")
            print(f"  First 200 chars: {preview['content'][:200]}...")

        return True
    else:
        print(f"❌ Preview failed: {response.text}")
        return False


def test_download_file(session_id: str, filename: str):
    """Test file download."""
    print(f"\nTesting file download for {filename}...")

    response = requests.get(
        f"{BASE_URL}/api/download/{session_id}/{filename}", stream=True
    )

    if response.status_code == 200:
        # Download first chunk to verify
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > 8192:  # Just test first chunk
                break

        print("✅ File download started:")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Content-Length: {response.headers.get('Content-Length')}")
        print(f"  Downloaded bytes: {total_size}")

        return True
    else:
        print(f"❌ Download failed: {response.text}")
        return False


def test_download_zip(session_id: str):
    """Test ZIP download."""
    print(f"\nTesting ZIP download for session {session_id}...")

    response = requests.get(f"{BASE_URL}/api/download/{session_id}/zip", stream=True)

    if response.status_code == 200:
        # Download first chunk
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > 8192:
                break

        print("✅ ZIP download started:")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Downloaded bytes: {total_size}")

        return True
    else:
        print(f"❌ ZIP download failed: {response.text}")
        return False


def test_download_token(session_id: str):
    """Test download token creation."""
    print(f"\nTesting download token creation...")

    response = requests.post(
        f"{BASE_URL}/api/download/token/{session_id}", params={"expires_in_hours": 1}
    )

    if response.status_code == 200:
        token_data = response.json()
        print("✅ Download token created:")
        print(f"  Token: {token_data['token'][:20]}...")
        print(f"  Expires at: {token_data['expires_at']}")

        # Validate token
        response = requests.post(
            f"{BASE_URL}/api/download/validate-token",
            params={"token": token_data["token"]},
        )

        if response.status_code == 200:
            validation = response.json()
            print("✅ Token validated:")
            print(f"  Valid: {validation['valid']}")
            print(f"  Session ID: {validation['session_id']}")
            print(f"  Remaining downloads: {validation['remaining_downloads']}")

        return True, token_data["token"]
    else:
        print(f"❌ Token creation failed: {response.text}")
        return False, None


def test_download_link(session_id: str, filename: str):
    """Test download link creation."""
    print(f"\nTesting download link creation for {filename}...")

    response = requests.post(
        f"{BASE_URL}/api/download/link/{session_id}/{filename}",
        params={"expires_in_hours": 1},
    )

    if response.status_code == 200:
        link_data = response.json()
        print("✅ Download link created:")
        print(f"  URL: {link_data['url']}")
        print(f"  Expires at: {link_data['expires_at']}")

        return True
    else:
        print(f"❌ Link creation failed: {response.text}")
        return False


def test_download_analytics():
    """Test download analytics."""
    print("\nTesting download analytics...")

    response = requests.get(f"{BASE_URL}/api/results/analytics/downloads")

    if response.status_code == 200:
        analytics = response.json()
        print("✅ Download analytics:")
        print(f"  Total downloads: {analytics.get('total_downloads', 0)}")
        print(f"  Success rate: {analytics.get('success_rate', 0):.1f}%")
        print(
            f"  Average download time: {analytics.get('average_download_time', 0):.2f}s"
        )

        if analytics.get("popular_files"):
            print("  Popular files:")
            for filename, count in analytics["popular_files"][:3]:
                print(f"    - {filename}: {count} downloads")

        return True
    else:
        print(f"❌ Analytics failed: {response.text}")
        return False


def test_results_summary():
    """Test results summary."""
    print("\nTesting results summary...")

    response = requests.get(f"{BASE_URL}/api/results/stats/summary")

    if response.status_code == 200:
        summary = response.json()
        print("✅ Results summary:")
        print(f"  Total sessions: {summary.get('total_sessions', 0)}")
        print(f"  Total files: {summary.get('total_files', 0)}")
        print(f"  Total size: {summary.get('total_size_mb', 0):.2f} MB")
        print(
            f"  Average files per session: {summary.get('average_files_per_session', 0):.1f}"
        )
        print(
            f"  Average processing time: {summary.get('average_processing_time', 0):.2f}s"
        )

        return True
    else:
        print(f"❌ Summary failed: {response.text}")
        return False


def test_download_manifest(session_id: str):
    """Test download manifest creation."""
    print(f"\nTesting download manifest for session {session_id}...")

    response = requests.post(f"{BASE_URL}/api/results/{session_id}/manifest")

    if response.status_code == 200:
        manifest = response.json()
        print("✅ Download manifest created:")
        print(f"  Manifest ID: {manifest['manifest_id']}")
        print(f"  Total files: {manifest['total_files']}")
        print(f"  Total size: {manifest['total_size']} bytes")
        print(f"  Expires at: {manifest['expires_at']}")

        return True
    else:
        print(f"❌ Manifest creation failed: {response.text}")
        return False


def test_active_downloads():
    """Test active downloads monitoring."""
    print("\nTesting active downloads...")

    response = requests.get(f"{BASE_URL}/api/download/progress/active")

    if response.status_code == 200:
        downloads = response.json()
        print(f"✅ Active downloads: {len(downloads)}")

        for dl in downloads[:3]:
            print(
                f"  - {dl['filename']}: {dl['progress']:.1f}% "
                f"({dl['bytes_sent']}/{dl['total_bytes']} bytes)"
            )

        return True
    else:
        print(f"❌ Active downloads failed: {response.text}")
        return False


def main():
    """Run all Sprint 5 tests."""
    print("=" * 60)
    print("Sprint 5 API Tests - Results & Downloads")
    print("=" * 60)
    print()

    # Check if API is running
    try:
        response = requests.get(BASE_URL)
        print(f"API is running at {BASE_URL}")
        print()
    except requests.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("Please start the API with: python run_api.py")
        return 1

    # Run tests
    all_passed = True

    # Set up test session
    session_id = setup_test_session()
    if not session_id:
        print("❌ Failed to set up test session")
        return 1

    print(f"\n✅ Test session created: {session_id}")

    # Wait a bit for processing to complete
    print("Waiting for processing to fully complete...")
    time.sleep(3)

    # Test 1: Get results
    success, files = test_get_results(session_id)
    if not success:
        all_passed = False

    # Test 2: Search results
    if not test_search_results():
        all_passed = False

    # Test 3: File preview
    if files:
        if not test_file_preview(session_id, files[0]["filename"]):
            all_passed = False

    # Test 4: File download
    if files:
        if not test_download_file(session_id, files[0]["filename"]):
            all_passed = False

    # Test 5: ZIP download
    if not test_download_zip(session_id):
        all_passed = False

    # Test 6: Download token
    success, token = test_download_token(session_id)
    if not success:
        all_passed = False

    # Test 7: Download link
    if files:
        if not test_download_link(session_id, files[0]["filename"]):
            all_passed = False

    # Test 8: Download analytics
    if not test_download_analytics():
        all_passed = False

    # Test 9: Results summary
    if not test_results_summary():
        all_passed = False

    # Test 10: Download manifest
    if not test_download_manifest(session_id):
        all_passed = False

    # Test 11: Active downloads
    if not test_active_downloads():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All Sprint 5 tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
