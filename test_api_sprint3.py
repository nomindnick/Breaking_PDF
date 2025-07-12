#!/usr/bin/env python3
"""
Test script for Sprint 3 API implementation.

Tests split proposal management, modifications, and execution.
"""
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

# API base URL
BASE_URL = "http://localhost:8000"


def setup_test_session() -> Tuple[Optional[str], Optional[str]]:
    """Set up a test session with processed PDF."""
    print("Setting up test session...")

    # Find test PDF
    test_file = None
    for f in [Path("Test_PDF_Set_1.pdf"), Path("Test_PDF_Set_2_ocr.pdf")]:
        if f.exists():
            test_file = f
            break

    if not test_file:
        print("❌ No test PDF file found")
        return None, None

    # Upload file
    print(f"Uploading {test_file.name}...")
    with open(test_file, "rb") as f:
        files = {"file": (test_file.name, f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return None, None

    file_id = response.json()["upload_id"]

    # Start processing
    print("Starting processing...")
    response = requests.post(f"{BASE_URL}/api/process", json={"file_id": file_id})

    if response.status_code != 200:
        print(f"❌ Processing failed: {response.text}")
        return None, None

    session_id = response.json()["session_id"]

    # Wait for processing to complete
    print("Waiting for processing to complete...")
    for i in range(30):  # Max 30 seconds
        response = requests.get(f"{BASE_URL}/api/process/{session_id}/status")
        if response.status_code == 200:
            status = response.json()["status"]
            if status == "confirmed":
                print("✅ Processing complete")
                return session_id, file_id
            elif status == "cancelled":
                print("❌ Processing failed")
                return None, None
        time.sleep(1)

    print("⏱️ Processing timeout")
    return session_id, file_id


def test_get_proposal(session_id: str):
    """Test getting split proposal."""
    print(f"\nTesting get split proposal for session {session_id}...")

    response = requests.get(f"{BASE_URL}/api/splits/{session_id}/proposal")

    if response.status_code == 200:
        data = response.json()
        print("✅ Proposal retrieved:")
        print(f"  Total pages: {data['total_pages']}")
        print(f"  Total segments: {data['total_segments']}")
        print(f"  Created at: {data['created_at']}")

        if data["segments"]:
            print("  Segments:")
            for seg in data["segments"][:3]:  # Show first 3
                print(
                    f"    - Pages {seg['start_page']}-{seg['end_page']}: "
                    f"{seg['document_type']} ({seg['confidence']:.2f})"
                )

        return True, data["segments"]
    else:
        print(f"❌ Get proposal failed: {response.text}")
        return False, []


def test_merge_segments(session_id: str, segments: list):
    """Test merging segments."""
    print(f"\nTesting segment merge...")

    if len(segments) < 2:
        print("⚠️ Not enough segments to test merge")
        return True

    # Try to merge first two segments
    segment_ids = [segments[0]["segment_id"], segments[1]["segment_id"]]

    response = requests.post(
        f"{BASE_URL}/api/splits/{session_id}/merge", json={"segment_ids": segment_ids}
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ Segments merged successfully")
        print(f"  New segment count: {data['total_segments']}")
        return True
    else:
        print(f"❌ Merge failed: {response.text}")
        return False


def test_split_segment(session_id: str, segments: list):
    """Test splitting a segment."""
    print(f"\nTesting segment split...")

    # Find a segment with multiple pages
    multi_page_segment = None
    for seg in segments:
        if seg["page_count"] > 1:
            multi_page_segment = seg
            break

    if not multi_page_segment:
        print("⚠️ No multi-page segment found to split")
        return True

    # Split in the middle
    split_page = (
        multi_page_segment["start_page"] + multi_page_segment["page_count"] // 2
    )

    response = requests.post(
        f"{BASE_URL}/api/splits/{session_id}/split",
        json={
            "segment_id": multi_page_segment["segment_id"],
            "page_number": split_page,
        },
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Segment split successfully at page {split_page}")
        print(f"  New segment count: {data['total_segments']}")
        return True
    else:
        print(f"❌ Split failed: {response.text}")
        return False


def test_update_segment(session_id: str, segments: list):
    """Test updating segment metadata."""
    print(f"\nTesting segment update...")

    if not segments:
        print("⚠️ No segments to update")
        return True

    segment_id = segments[0]["segment_id"]

    response = requests.patch(
        f"{BASE_URL}/api/splits/{session_id}/segments/{segment_id}",
        json={
            "document_type": "invoice",
            "metadata": {"test": "value"},
            "confidence": 0.95,
        },
    )

    if response.status_code == 200:
        print("✅ Segment updated successfully")
        return True
    else:
        print(f"❌ Update failed: {response.text}")
        return False


def test_segment_preview(session_id: str, segments: list):
    """Test segment preview generation."""
    print(f"\nTesting segment preview...")

    if not segments:
        print("⚠️ No segments to preview")
        return True

    segment_id = segments[0]["segment_id"]

    response = requests.get(
        f"{BASE_URL}/api/splits/{session_id}/preview/{segment_id}?max_pages=2"
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ Preview generated:")
        print(f"  Pages included: {data['pages_included']}")
        print(f"  Images returned: {len(data['images'])}")

        # Check if images are valid base64
        for i, img in enumerate(data["images"]):
            if img.startswith("data:image/png;base64,"):
                try:
                    base64.b64decode(img.split(",")[1])
                    print(f"  Image {i+1}: Valid")
                except:
                    print(f"  Image {i+1}: Invalid base64")
            else:
                print(f"  Image {i+1}: Invalid format")

        return True
    else:
        print(f"❌ Preview failed: {response.text}")
        return False


def test_execute_split(session_id: str):
    """Test split execution."""
    print(f"\nTesting split execution for session {session_id}...")

    response = requests.post(f"{BASE_URL}/api/splits/{session_id}/execute")

    if response.status_code == 200:
        data = response.json()
        split_id = data["split_id"]
        print(f"✅ Split execution started: {split_id}")

        # Wait for completion
        print("Waiting for split to complete...")
        for i in range(30):  # Max 30 seconds
            response = requests.get(f"{BASE_URL}/api/sessions/{session_id}")
            if response.status_code == 200:
                status = response.json()["status"]
                if status == "complete":
                    print("✅ Split completed")
                    return True, split_id
                elif status == "cancelled":
                    print("❌ Split failed")
                    return False, None
            time.sleep(1)

        print("⏱️ Split timeout")
        return False, split_id
    else:
        print(f"❌ Split execution failed: {response.text}")
        return False, None


def test_get_results(session_id: str):
    """Test getting split results."""
    print(f"\nTesting split results for session {session_id}...")

    response = requests.get(f"{BASE_URL}/api/splits/{session_id}/results")

    if response.status_code == 200:
        data = response.json()
        print("✅ Results retrieved:")
        print(f"  Files created: {data['files_created']}")
        print(f"  Processing time: {data['processing_time']:.2f}s")
        print(f"  Output size: {data['output_size_bytes']} bytes")

        if data["output_files"]:
            print("  Output files:")
            for file_info in data["output_files"][:5]:  # Show first 5
                print(f"    - {file_info['filename']} ({file_info['size']} bytes)")

        return True, data["output_files"]
    else:
        print(f"❌ Get results failed: {response.text}")
        return False, []


def test_download_file(session_id: str, files: list):
    """Test file download."""
    print(f"\nTesting file download...")

    if not files:
        print("⚠️ No files to download")
        return True

    filename = files[0]["filename"]

    response = requests.get(f"{BASE_URL}/api/splits/{session_id}/download/{filename}")

    if response.status_code == 200:
        print(f"✅ File downloaded: {filename}")
        print(f"  Content type: {response.headers.get('content-type')}")
        print(f"  Size: {len(response.content)} bytes")
        return True
    else:
        print(f"❌ Download failed: {response.text}")
        return False


def test_download_zip(session_id: str):
    """Test ZIP download."""
    print(f"\nTesting ZIP download...")

    response = requests.get(f"{BASE_URL}/api/splits/{session_id}/download/zip")

    if response.status_code == 200:
        print("✅ ZIP downloaded")
        print(f"  Content type: {response.headers.get('content-type')}")
        print(f"  Size: {len(response.content)} bytes")

        # Verify it's a valid ZIP
        try:
            import io
            import zipfile

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                print(f"  Files in ZIP: {len(zf.namelist())}")
                for name in zf.namelist()[:5]:  # Show first 5
                    print(f"    - {name}")
        except Exception as e:
            print(f"  ❌ Invalid ZIP: {e}")
            return False

        return True
    else:
        print(f"❌ ZIP download failed: {response.text}")
        return False


def main():
    """Run all Sprint 3 tests."""
    print("=" * 60)
    print("Sprint 3 API Tests")
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
    session_id, file_id = setup_test_session()
    if not session_id:
        print("❌ Failed to set up test session")
        return 1

    # Test proposal management
    success, segments = test_get_proposal(session_id)
    if not success:
        all_passed = False

    if segments:
        # Test modifications
        if not test_merge_segments(session_id, segments):
            all_passed = False

        # Get updated segments
        _, segments = test_get_proposal(session_id)

        if not test_split_segment(session_id, segments):
            all_passed = False

        if not test_update_segment(session_id, segments):
            all_passed = False

        if not test_segment_preview(session_id, segments):
            all_passed = False

    # Test split execution
    success, split_id = test_execute_split(session_id)
    if not success:
        all_passed = False
    else:
        # Test results
        success, files = test_get_results(session_id)
        if not success:
            all_passed = False

        if files:
            if not test_download_file(session_id, files):
                all_passed = False

            if not test_download_zip(session_id):
                all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All Sprint 3 tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
