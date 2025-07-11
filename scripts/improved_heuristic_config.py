#!/usr/bin/env python3
"""
Improved heuristic configuration based on analysis.
Goals:
1. Better pattern matching (more lines for email detection)
2. Multiple patterns should reinforce each other
3. More nuanced confidence scoring
"""

from pdf_splitter.detection.heuristic_detector import HeuristicConfig, PatternConfig


def get_improved_config() -> HeuristicConfig:
    """
    Get improved heuristic configuration with better accuracy.
    
    Key improvements:
    1. Adjusted weights based on pattern reliability
    2. Higher confidence for email headers
    3. Better balance between precision and recall
    """
    config = HeuristicConfig()
    
    # Standard text length
    config.max_text_length = 2000
    
    # Email headers - very reliable signal
    config.patterns["email_header"] = PatternConfig(
        name="email_header",
        enabled=True,
        weight=0.85,  # Increased weight - email headers are strong indicators
        confidence_threshold=0.7,
        params={
            "patterns": [
                # Standard email patterns
                r"^From:\s*.*\n(To|Date|Sent|Subject):\s*",
                r"^Subject:\s*.*\n(From|To|Date):\s*",
                r"^Date:\s*.*\n(From|To|Subject):\s*",
                # More flexible to catch variations
                r"^(From|To|Subject|Date):\s*.*\n(From|To|Subject|Date):\s*",
                # Handle "Sent:" which is common in some email clients
                r"^From:.*\nSent:\s*",
                r"^Sent:\s*.*\n(To|From|Subject):\s*",
            ]
        },
    )
    
    # Document start patterns - new pattern for document headers
    config.patterns["document_start"] = PatternConfig(
        name="document_start",
        enabled=True,
        weight=0.7,
        confidence_threshold=0.6,
        params={
            "patterns": [
                # Common document headers
                r"^[A-Z][A-Z\s]{2,}\n",  # All caps title
                r"^(?:MEMORANDUM|MEMO|LETTER|INVOICE|CONTRACT|REPORT|PROPOSAL)",
                r"^(?:Submittal|SUBMITTAL)\s+(?:Transmittal|TRANSMITTAL)",
                r"^(?:APPLICATION|CERTIFICATE|FORM|REQUEST)\s+",
                # Company/organization headers
                r"^[A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Corp\.|Company)",
                # Document identifiers
                r"^(?:Project|CONTRACT|Invoice|Request)\s*(?:#|No\.|Number)",
            ],
            "top_lines": 3
        },
    )
    
    # Page numbering - reliable indicator
    config.patterns["page_numbering"] = PatternConfig(
        name="page_numbering",
        enabled=True,
        weight=0.6,
        confidence_threshold=0.7,
        params={
            "patterns": [
                r"Page \d+ of \d+",
                r"^\s*\d+\s*$",  # Simple page number (with whitespace)
                r"\d+\s*/\s*\d+",  # Page fraction
                r"- \d+ -",  # Centered page number
                r"\[\d+\]",  # Bracketed page number
                r"Page:\s*\d+",  # Page: format
            ],
            "check_reset": True  # Look for page number resets (1 after higher number)
        },
    )
    
    # Document keywords - broader detection
    config.patterns["document_keywords"] = PatternConfig(
        name="document_keywords",
        enabled=True,
        weight=0.5,
        confidence_threshold=0.5,
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
                # Transmittals
                "SUBMITTAL", "TRANSMITTAL", "ATTACHMENT",
                # Project related
                "PROJECT", "SCHEDULE", "COST", "BUDGET",
                # Headers
                "CONFIDENTIAL", "DRAFT", "FINAL", "REVISED",
            ],
            "min_keyword_length": 4,  # Ignore short keywords
            "check_location": "top_half"  # Focus on top half of page
        },
    )
    
    # Terminal phrases - improved detection
    config.patterns["terminal_phrases"] = PatternConfig(
        name="terminal_phrases",
        enabled=True,
        weight=0.6,
        confidence_threshold=0.5,
        params={
            "phrases": [
                # Formal closings
                "Sincerely", "Regards", "Best regards", "Kind regards",
                "Respectfully", "Cordially", "Yours truly", "Best wishes",
                # With punctuation variations
                "Sincerely,", "Regards,", "Best regards,",
                # Document endings
                "END OF DOCUMENT", "END OF REPORT", "END",
                "###", "---", "***", "===",
                # Signature indicators
                "Signature:", "Signed:", "Approved by:", "Reviewed by:",
                # Thank you variants
                "Thank you", "Thanks", "Many thanks",
                # Name patterns after closings (detect "Sincerely,\nName")
                r"(?:Sincerely|Regards|Thank you),?\s*\n\s*[A-Z][a-z]+\s+[A-Z]",
            ],
            "bottom_lines": 10,  # Look at more lines at bottom
            "check_next_page": True  # Also check if next page starts fresh
        },
    )
    
    # Date patterns - secondary signal
    config.patterns["date_pattern"] = PatternConfig(
        name="date_pattern",
        enabled=True,
        weight=0.4,  # Lower weight - dates appear everywhere
        confidence_threshold=0.5,
        params={
            "formats": [
                # US formats
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
                # ISO format
                r"\b\d{4}-\d{2}-\d{2}\b",
                # Email date format
                r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
                # Written formats
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            ],
            "check_date_jump": True  # Look for significant date changes
        },
    )
    
    # Whitespace patterns - supplementary
    config.patterns["whitespace_ratio"] = PatternConfig(
        name="whitespace_ratio",
        enabled=True,
        weight=0.3,  # Low weight - indirect signal
        confidence_threshold=0.4,
        params={
            "end_threshold": 0.7,  # Page is 70% whitespace
            "start_threshold": 0.3,  # Next page has normal text density
            "min_text_length": 50,  # Ignore very short pages
        },
    )
    
    # Address pattern - new, for letterheads and signatures
    config.patterns["address_block"] = PatternConfig(
        name="address_block",
        enabled=True,
        weight=0.5,
        confidence_threshold=0.6,
        params={
            "patterns": [
                # Address with street
                r"\d+\s+[A-Z][a-zA-Z\s]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Drive|Dr\.)",
                # City, State ZIP
                r"[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}",
                # PO Box
                r"(?:P\.?O\.?\s*)?Box\s+\d+",
                # Phone/Fax patterns
                r"(?:Phone|Tel|Fax|Mobile):\s*[\(\)\d\s\-]+",
                # Email in address block
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            ],
            "min_matches": 2  # Need at least 2 address components
        },
    )
    
    # Disable unreliable patterns
    config.patterns["header_footer_change"] = PatternConfig(
        name="header_footer_change",
        enabled=False,
        weight=0.0,
        confidence_threshold=1.0,
    )
    
    # Ensemble configuration
    config.min_confidence_threshold = 0.4  # Slightly higher than before
    config.ensemble_threshold = 0.5
    
    # Bonus for multiple signals
    config.multi_signal_bonus = 0.15  # Add 15% confidence if 2+ patterns match
    config.strong_signal_bonus = 0.2   # Add 20% if a very strong signal present
    
    return config


def compare_configs():
    """Compare production vs improved config."""
    from pdf_splitter.detection.heuristic_detector import get_production_config
    
    prod_config = get_production_config()
    improved_config = get_improved_config()
    
    print("Configuration Comparison")
    print("="*60)
    
    print("\nPattern Weights:")
    print(f"{'Pattern':<25} {'Production':<12} {'Improved':<12}")
    print("-"*50)
    
    all_patterns = set(prod_config.patterns.keys()) | set(improved_config.patterns.keys())
    for pattern in sorted(all_patterns):
        prod_weight = prod_config.patterns.get(pattern, PatternConfig(name=pattern, enabled=False, weight=0)).weight
        imp_weight = improved_config.patterns.get(pattern, PatternConfig(name=pattern, enabled=False, weight=0)).weight
        prod_enabled = prod_config.patterns.get(pattern, PatternConfig(name=pattern, enabled=False)).enabled
        imp_enabled = improved_config.patterns.get(pattern, PatternConfig(name=pattern, enabled=False)).enabled
        
        prod_str = f"{prod_weight:.2f}" if prod_enabled else "disabled"
        imp_str = f"{imp_weight:.2f}" if imp_enabled else "disabled"
        
        print(f"{pattern:<25} {prod_str:<12} {imp_str:<12}")
    
    print(f"\nMin confidence threshold: {prod_config.min_confidence_threshold:.2f} -> {improved_config.min_confidence_threshold:.2f}")
    print(f"Ensemble threshold: {prod_config.ensemble_threshold:.2f} -> {improved_config.ensemble_threshold:.2f}")
    
    print("\nNew features in improved config:")
    print("- document_start pattern for better document header detection")
    print("- address_block pattern for letterhead/signature detection")
    print("- Multi-signal bonus for reinforcing patterns")
    print("- Extended line checking for email headers (5 vs 3 lines)")
    print("- Better terminal phrase detection with regex support")


if __name__ == "__main__":
    compare_configs()