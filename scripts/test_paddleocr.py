#!/usr/bin/env python3
"""Test script to verify PaddleOCR installation and functionality."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paddleocr import PaddleOCR


def test_paddleocr():
    """Test basic PaddleOCR functionality."""
    print("Initializing PaddleOCR...")
    
    # Initialize PaddleOCR with English language support
    # use_angle_cls=True enables text angle classification
    # lang='en' for English
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        show_log=False  # Reduce verbosity
    )
    
    print("✓ PaddleOCR initialized successfully")
    print(f"✓ Detection model: {ocr.text_detector is not None}")
    print(f"✓ Recognition model: {ocr.text_recognizer is not None}")
    print(f"✓ Angle classifier: {ocr.use_angle_cls}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_paddleocr()
        if success:
            print("\n✅ PaddleOCR is properly installed and configured!")
    except Exception as e:
        print(f"\n❌ Error testing PaddleOCR: {e}")
        sys.exit(1)