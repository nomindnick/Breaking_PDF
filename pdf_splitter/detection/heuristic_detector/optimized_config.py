"""
Optimized heuristic configuration based on experimental results.

This configuration is tuned for production use based on extensive testing
with real-world PDFs. It prioritizes high-accuracy patterns and minimizes
false positives.

Experimental Results Summary:
- Email headers: 100% accuracy (when detected)
- Page numbering: 100% accuracy (when detected)
- Terminal phrases: 50% accuracy
- Header/footer changes: 46.3% accuracy (causes many false positives)
- Date patterns: 25% accuracy
- Document keywords: 24.5% accuracy
"""

from pdf_splitter.detection.heuristic_detector import HeuristicConfig, PatternConfig


def get_optimized_config() -> HeuristicConfig:
    """
    Get optimized heuristic configuration for production use.

    This configuration is based on experimental results showing:
    - Default config: F1=0.533, Precision=0.375, Recall=0.923
    - Need to reduce false positives while maintaining good recall

    Returns:
        Optimized HeuristicConfig
    """
    config = HeuristicConfig()

    # Email header pattern - 100% accuracy, highest weight
    config.patterns["email_header"] = PatternConfig(
        name="email_header",
        enabled=True,
        weight=1.0,  # Maximum weight for perfect accuracy
        confidence_threshold=0.85,  # High threshold for high confidence
        params={
            "patterns": [
                r"^From:\s*.*\nTo:\s*.*\nSubject:\s*",
                r"^From:\s*.*\nSent:\s*.*\nTo:\s*",
                r"^From:\s*.*\nDate:\s*.*\nTo:\s*",  # Added variation
            ]
        },
    )

    # Page numbering pattern - 100% accuracy, high weight
    config.patterns["page_numbering"] = PatternConfig(
        name="page_numbering",
        enabled=True,
        weight=0.95,  # Very high weight for perfect accuracy
        confidence_threshold=0.8,
        params={
            "patterns": [
                r"Page \d+ of \d+",
                r"Page \d+$",
                r"^\d+$",
                r"\d+/\d+$",  # Added page fraction pattern
            ]
        },
    )

    # Document keywords - 24.5% accuracy, moderate weight
    config.patterns["document_keywords"] = PatternConfig(
        name="document_keywords",
        enabled=True,
        weight=0.6,  # Reduced from 0.8 due to lower accuracy
        confidence_threshold=0.7,
        params={
            "keywords": [
                # High-confidence keywords based on test data
                "MEMORANDUM",
                "INVOICE",
                "CONTRACT",
                "AGREEMENT",
                "REQUEST FOR INFORMATION",
                "RFI",
                "SUBMITTAL",
                "APPLICATION FOR PAYMENT",
                # Additional common document types
                "PURCHASE ORDER",
                "QUOTATION",
                "PROPOSAL",
            ]
        },
    )

    # Terminal phrases - 50% accuracy, lower weight
    config.patterns["terminal_phrases"] = PatternConfig(
        name="terminal_phrases",
        enabled=True,
        weight=0.4,  # Reduced from 0.6 due to 50% accuracy
        confidence_threshold=0.6,
        params={
            "phrases": [
                "Sincerely",
                "Very truly yours",
                "Respectfully",
                "Best regards",
                "Thank you",
                "Regards",
                "Respectfully submitted",
                "Yours truly",
            ]
        },
    )

    # Date pattern - 25% accuracy, minimal weight
    config.patterns["date_pattern"] = PatternConfig(
        name="date_pattern",
        enabled=True,
        weight=0.3,  # Significantly reduced from 0.7 due to low accuracy
        confidence_threshold=0.7,
        params={
            "formats": [
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
                r"\b\w+ \d{1,2}, \d{4}\b",
                r"\b\d{4}-\d{2}-\d{2}\b",  # ISO format
            ]
        },
    )

    # Whitespace ratio - Keep but with low weight as supplementary signal
    config.patterns["whitespace_ratio"] = PatternConfig(
        name="whitespace_ratio",
        enabled=True,
        weight=0.3,  # Reduced from 0.5
        confidence_threshold=0.5,
        params={
            "end_threshold": 0.7,  # Increased from 0.6 for fewer false positives
            "start_threshold": 0.3,
        },
    )

    # Header/footer change - DISABLED due to excessive false positives
    config.patterns["header_footer_change"] = PatternConfig(
        name="header_footer_change",
        enabled=False,  # Disabled - 46.3% accuracy causes too many false positives
        weight=0.0,
        confidence_threshold=1.0,  # Effectively disabled
        params={
            "top_lines": 3,
            "bottom_lines": 3,
        },
    )

    # Adjust global thresholds for better precision
    config.min_confidence_threshold = 0.4  # Increased from 0.3
    config.ensemble_threshold = 0.6  # Increased from 0.5

    return config


def get_fast_screen_config() -> HeuristicConfig:
    """
    Get configuration optimized for fast screening (high recall).

    Use this when heuristics are the first pass in a cascade,
    where false positives will be filtered by subsequent detectors.

    Returns:
        HeuristicConfig optimized for high recall
    """
    config = get_optimized_config()

    # Lower thresholds for higher recall
    config.min_confidence_threshold = 0.3
    config.ensemble_threshold = 0.4

    # Re-enable header/footer with very low weight
    config.patterns["header_footer_change"].enabled = True
    config.patterns["header_footer_change"].weight = 0.2

    return config


def get_high_precision_config() -> HeuristicConfig:
    """
    Get configuration optimized for high precision (fewer false positives).

    Use this when heuristics are the only detector or when false
    positives are very costly.

    Returns:
        HeuristicConfig optimized for high precision
    """
    config = HeuristicConfig()

    # Only enable the most accurate patterns
    config.patterns["email_header"] = PatternConfig(
        name="email_header",
        enabled=True,
        weight=1.0,
        confidence_threshold=0.9,
        params={
            "patterns": [
                r"^From:\s*.*\nTo:\s*.*\nSubject:\s*",
                r"^From:\s*.*\nSent:\s*.*\nTo:\s*",
            ]
        },
    )

    config.patterns["page_numbering"] = PatternConfig(
        name="page_numbering",
        enabled=True,
        weight=1.0,
        confidence_threshold=0.85,
        params={
            "patterns": [
                r"Page \d+ of \d+",
                r"Page \d+$",
            ]
        },
    )

    # Disable all other patterns
    for pattern_name in [
        "document_keywords",
        "terminal_phrases",
        "date_pattern",
        "whitespace_ratio",
        "header_footer_change",
    ]:
        config.patterns[pattern_name].enabled = False

    # High thresholds for very high precision
    config.min_confidence_threshold = 0.6
    config.ensemble_threshold = 0.8

    return config
