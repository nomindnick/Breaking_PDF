"""Production boundary detector factory."""

from pdf_splitter.detection import EmbeddingsDetector


def create_production_detector():
    """
    Create the production boundary detector.

    Returns an EmbeddingsDetector with optimal settings:
    - Model: all-MiniLM-L6-v2
    - Threshold: 0.5
    - Expected F1: ~0.65-0.70
    """
    return EmbeddingsDetector(model_name="all-MiniLM-L6-v2", similarity_threshold=0.5)
