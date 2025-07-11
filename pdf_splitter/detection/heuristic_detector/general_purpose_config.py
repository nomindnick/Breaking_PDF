"""
General-purpose heuristic configuration for production use.

This configuration is designed to work with diverse document types without
overfitting to specific test data. It uses balanced weights and conservative
thresholds to ensure the system generalizes well.

Key principles:
1. No pattern is weighted significantly higher than others
2. Conservative confidence thresholds to trigger LLM verification
3. Broad pattern definitions that work across document types
4. Designed to be the first stage in a cascade with LLM verification
"""

from pdf_splitter.detection.heuristic_detector import HeuristicConfig, PatternConfig


def get_general_purpose_config() -> HeuristicConfig:
    """
    Get general-purpose heuristic configuration for production use.
    
    This configuration:
    - Uses balanced weights (no favoritism based on test data)
    - Sets conservative thresholds to ensure LLM verification
    - Includes broad patterns that work across document types
    - Optimized for use with cascade strategy
    
    Returns:
        General-purpose HeuristicConfig
    """
    config = HeuristicConfig()
    
    # Base weight for most patterns - balanced approach
    base_weight = 0.5
    
    # Email headers - common in business documents but not universal
    config.patterns["email_header"] = PatternConfig(
        name="email_header",
        enabled=True,
        weight=base_weight * 1.2,  # Slightly higher but not dominant
        confidence_threshold=0.6,  # Conservative threshold
        params={
            "patterns": [
                r"^From:\s*.*\n(To|Date|Sent|Subject):\s*",
                r"^Subject:\s*.*\n(From|To|Date):\s*",
                r"^Date:\s*.*\n(From|To|Subject):\s*",
                # More flexible email patterns
                r"^(From|To|Subject|Date):\s*.*\n(From|To|Subject|Date):\s*",
            ]
        },
    )
    
    # Page numbering - reliable when present but not always there
    config.patterns["page_numbering"] = PatternConfig(
        name="page_numbering",
        enabled=True,
        weight=base_weight * 1.1,  # Slightly higher weight
        confidence_threshold=0.6,
        params={
            "patterns": [
                r"Page \d+ of \d+",
                r"^\d+$",  # Simple page number
                r"\d+\s*/\s*\d+",  # Page fraction
                r"- \d+ -",  # Centered page number
                r"\[\d+\]",  # Bracketed page number
            ]
        },
    )
    
    # Document keywords - broad set for various document types
    config.patterns["document_keywords"] = PatternConfig(
        name="document_keywords",
        enabled=True,
        weight=base_weight,
        confidence_threshold=0.5,  # Lower threshold for diversity
        params={
            "keywords": [
                # Business documents
                "MEMORANDUM", "MEMO", "LETTER", "CORRESPONDENCE",
                "INVOICE", "RECEIPT", "STATEMENT", "BILL",
                "CONTRACT", "AGREEMENT", "TERMS", "CONDITIONS",
                "PROPOSAL", "QUOTATION", "ESTIMATE", "BID",
                # Reports and forms
                "REPORT", "SUMMARY", "ANALYSIS", "REVIEW",
                "APPLICATION", "FORM", "REQUEST", "CLAIM",
                "CERTIFICATE", "NOTICE", "ANNOUNCEMENT",
                # Legal/official
                "EXHIBIT", "ADDENDUM", "APPENDIX", "ATTACHMENT",
                "RESOLUTION", "MINUTES", "AGENDA", "POLICY",
                # Headers
                "CONFIDENTIAL", "DRAFT", "FINAL", "REVISED",
                "INTERNAL", "EXTERNAL", "PUBLIC", "PRIVATE",
            ]
        },
    )
    
    # Terminal phrases - common document endings
    config.patterns["terminal_phrases"] = PatternConfig(
        name="terminal_phrases",
        enabled=True,
        weight=base_weight * 0.8,  # Slightly lower - can be ambiguous
        confidence_threshold=0.5,
        params={
            "phrases": [
                # Formal closings
                "Sincerely", "Regards", "Best regards", "Kind regards",
                "Respectfully", "Cordially", "Yours truly", "Best wishes",
                # Document endings
                "END OF DOCUMENT", "END OF REPORT", "END",
                "###", "---", "***",
                # Signature blocks
                "Signature:", "Signed:", "Approved by:", "Reviewed by:",
                "Prepared by:", "Submitted by:",
                # Thank you variants
                "Thank you", "Thanks", "Many thanks",
            ]
        },
    )
    
    # Date patterns - common but appear throughout documents
    config.patterns["date_pattern"] = PatternConfig(
        name="date_pattern",
        enabled=True,
        weight=base_weight * 0.6,  # Lower weight - dates are everywhere
        confidence_threshold=0.5,
        params={
            "formats": [
                # US formats
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
                # ISO format
                r"\b\d{4}-\d{2}-\d{2}\b",
                # Written formats
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
                r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            ]
        },
    )
    
    # Whitespace patterns - supplementary signal
    config.patterns["whitespace_ratio"] = PatternConfig(
        name="whitespace_ratio",
        enabled=True,
        weight=base_weight * 0.5,  # Lower weight - indirect signal
        confidence_threshold=0.4,
        params={
            "end_threshold": 0.6,  # More conservative than before
            "start_threshold": 0.4,  # More conservative
        },
    )
    
    # Header/footer changes - DISABLED due to too many false positives
    # This pattern is too noisy for general use
    config.patterns["header_footer_change"] = PatternConfig(
        name="header_footer_change",
        enabled=False,  # Disabled - causes too many false positives
        weight=0.0,
        confidence_threshold=1.0,
        params={
            "top_lines": 3,
            "bottom_lines": 3,
            "similarity_threshold": 0.9,  # Would need very high threshold
        },
    )
    
    # Global thresholds - conservative to ensure LLM verification
    config.min_confidence_threshold = 0.3  # Accept weak signals
    config.ensemble_threshold = 0.5  # Moderate ensemble threshold
    
    # This configuration is designed to:
    # 1. Cast a wide net (low min_confidence_threshold)
    # 2. Produce moderate confidence scores (0.4-0.7 typically)
    # 3. Trigger LLM verification for most boundaries
    # 4. Work well with diverse document types
    
    return config


def get_production_config() -> HeuristicConfig:
    """
    Alias for get_general_purpose_config.
    Use this in production environments.
    """
    return get_general_purpose_config()


def get_conservative_config() -> HeuristicConfig:
    """
    Get a more conservative configuration that relies heavily on LLM.
    
    This configuration:
    - Has even lower confidence thresholds
    - Triggers LLM verification for almost all detections
    - Maximizes accuracy at the cost of more LLM calls
    
    Returns:
        Conservative HeuristicConfig
    """
    config = get_general_purpose_config()
    
    # Lower all weights to produce lower confidence scores
    for pattern in config.patterns.values():
        pattern.weight *= 0.7
        pattern.confidence_threshold *= 0.8
    
    # Even more conservative thresholds
    config.min_confidence_threshold = 0.2
    config.ensemble_threshold = 0.4
    
    return config