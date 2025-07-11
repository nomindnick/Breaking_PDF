"""
Improved heuristic configuration based on accuracy analysis.

This configuration addresses the issues found in testing:
1. Many false negatives (missed boundaries) 
2. Single-pattern detections are unreliable
3. Email patterns not catching all variations
"""

from pdf_splitter.detection.heuristic_detector import HeuristicConfig, PatternConfig


def get_improved_config() -> HeuristicConfig:
    """
    Get improved heuristic configuration for better accuracy.
    
    Changes from production config:
    1. Higher weights for reliable patterns (email, terminal phrases)
    2. Added more email pattern variations
    3. Better document keyword coverage
    4. Tuned thresholds based on test results
    """
    config = HeuristicConfig()
    
    # Email headers - most reliable signal
    config.patterns["email_header"] = PatternConfig(
        name="email_header",
        enabled=True,
        weight=0.9,  # Very high weight - emails are strong boundaries
        confidence_threshold=0.7,
        params={
            "patterns": [
                # Standard patterns
                r"^From:\s*.*\n(To|Date|Sent|Subject):\s*",
                r"^Subject:\s*.*\n(From|To|Date):\s*",
                r"^Date:\s*.*\n(From|To|Subject):\s*",
                # Sent: pattern (common in Outlook)
                r"^From:.*\nSent:\s*",
                r"^Sent:\s*.*\n(To|From|Subject):\s*",
                # More flexible patterns
                r"^(From|To|Subject|Date|Sent):\s*.*\n(From|To|Subject|Date|Sent):\s*",
                # CC/BCC patterns
                r"^(From|To):\s*.*\n(CC|BCC):\s*",
            ]
        },
    )
    
    # Terminal phrases - good end-of-document indicator
    config.patterns["terminal_phrases"] = PatternConfig(
        name="terminal_phrases",
        enabled=True,
        weight=0.75,  # Higher weight than production
        confidence_threshold=0.6,
        params={
            "phrases": [
                # Formal closings
                "Sincerely", "Regards", "Best regards", "Kind regards",
                "Respectfully", "Cordially", "Yours truly", "Best wishes",
                # Thank you variants (very common)
                "Thank you", "Thanks", "Many thanks", "Thank You",
                # Document endings
                "END OF DOCUMENT", "END OF REPORT", "END",
                # Separators
                "###", "---", "***",
                # Signature blocks
                "Signature:", "Signed:", "Approved by:", "Reviewed by:",
                "Prepared by:", "Submitted by:",
                # Name patterns after thank you
                "Thank you,", "Thanks,", "Sincerely,", "Regards,",
            ]
        },
    )
    
    # Document keywords - broader set
    config.patterns["document_keywords"] = PatternConfig(
        name="document_keywords",
        enabled=True,
        weight=0.65,  # Higher than production
        confidence_threshold=0.5,
        params={
            "keywords": [
                # Business documents
                "MEMORANDUM", "MEMO", "LETTER", "CORRESPONDENCE",
                "INVOICE", "RECEIPT", "STATEMENT", "BILL",
                "CONTRACT", "AGREEMENT", "TERMS", "CONDITIONS",
                "PROPOSAL", "QUOTATION", "ESTIMATE", "BID",
                # Forms and applications
                "APPLICATION", "FORM", "REQUEST", "CLAIM",
                "CERTIFICATE", "NOTICE", "ANNOUNCEMENT",
                # Reports
                "REPORT", "SUMMARY", "ANALYSIS", "REVIEW",
                # Transmittals (very common in test data)
                "SUBMITTAL", "TRANSMITTAL", "ATTACHMENT",
                "EXHIBIT", "ADDENDUM", "APPENDIX",
                # Project documents
                "PROJECT", "SCHEDULE", "COST", "BUDGET",
                "CONTINUATION", "SHEET", "PAGE",
                # Request for Information
                "RFI", "REQUEST FOR INFORMATION",
                # Headers
                "CONFIDENTIAL", "DRAFT", "FINAL", "REVISED",
            ]
        },
    )
    
    # Page numbering - reliable when it resets
    config.patterns["page_numbering"] = PatternConfig(
        name="page_numbering",
        enabled=True,
        weight=0.7,  # Higher weight
        confidence_threshold=0.6,
        params={
            "patterns": [
                r"Page \d+ of \d+",
                r"^\d+$",  # Simple page number
                r"\d+\s*/\s*\d+",  # Page fraction
                r"- \d+ -",  # Centered page number
                r"\[\d+\]",  # Bracketed page number
                r"Page:\s*\d+",  # Page: X format
                r"Sheet\s+\d+\s+of\s+\d+",  # Sheet X of Y
            ]
        },
    )
    
    # Date patterns - lower weight but still useful
    config.patterns["date_pattern"] = PatternConfig(
        name="date_pattern",
        enabled=True,
        weight=0.45,  # Slightly higher than production
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
                # Email date format
                r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
                # Time stamps
                r"\d{1,2}:\d{2}:\d{2}\s*(AM|PM|am|pm)?",
            ]
        },
    )
    
    # Whitespace ratio - keep but with adjusted thresholds
    config.patterns["whitespace_ratio"] = PatternConfig(
        name="whitespace_ratio",
        enabled=True,
        weight=0.4,  # Slightly higher
        confidence_threshold=0.4,
        params={
            "end_threshold": 0.65,  # Slightly less strict
            "start_threshold": 0.35,  # More lenient
        },
    )
    
    # Header/footer changes - keep disabled
    config.patterns["header_footer_change"] = PatternConfig(
        name="header_footer_change",
        enabled=False,
        weight=0.0,
        confidence_threshold=1.0,
    )
    
    # Global thresholds - adjusted for better recall
    config.min_confidence_threshold = 0.35  # Slightly higher than production
    config.ensemble_threshold = 0.45  # Lower to catch more boundaries
    
    return config