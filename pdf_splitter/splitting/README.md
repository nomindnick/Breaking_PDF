# PDF Splitting Module

The splitting module provides intelligent PDF document separation with smart filename generation and session management.

## Overview

This module takes the output from the detection module (document boundaries) and splits a multi-document PDF into individual files with meaningful names.

## Key Components

### 1. Data Models (`models.py`)

- **DocumentSegment**: Represents a single document within a PDF
  ```python
  segment = DocumentSegment(
      start_page=1,
      end_page=5,
      document_type="invoice",
      confidence=0.85,
      suggested_filename="Invoice_2024-03-15_INV-12345.pdf"
  )
  ```

- **SplitProposal**: Container for all proposed segments
  ```python
  proposal = SplitProposal(
      pdf_path="/path/to/document.pdf",
      total_pages=100,
      segments=[segment1, segment2, ...]
  )
  ```

- **SplitSession**: Manages stateful splitting operations
  - States: draft → confirmed → processing → complete/cancelled
  - Tracks modifications and results
  - Automatic expiration after 24 hours

### 2. PDFSplitter Service (`pdf_splitter.py`)

The main service that handles document splitting with intelligent features:

#### Document Type Detection
Recognizes 12+ document types:
- Emails (with RE:/FW: detection)
- Invoices (with number extraction)
- Letters/Correspondence
- Reports and RFIs
- Contracts and Agreements
- Memos and Notices
- Meeting Minutes
- Forms and Applications

#### Smart Filename Generation
```python
# Examples of generated filenames:
"Email_2024-03-15_Project_Update.pdf"
"Invoice_2024-03-20_INV-12345.pdf"
"Letter_2024-03-10_Insurance_Claim.pdf"
"Report_2024-03-25_Quarterly_Review.pdf"
```

Features:
- Extracts relevant dates (newest by default)
- Finds unique identifiers (invoice numbers, case numbers)
- Creates content summaries for generic documents
- Sanitizes filenames for filesystem safety
- Ensures unique paths (adds counter if needed)

#### Usage Example
```python
from pdf_splitter.splitting import PDFSplitter
from pdf_splitter.detection import ProcessedPage

# Create splitter instance
splitter = PDFSplitter()

# Generate proposal from detection results
pages = [ProcessedPage(...), ...]  # From detection module
proposal = splitter.generate_proposal(
    pdf_path="/path/to/multi_doc.pdf",
    pages=pages
)

# Review and modify proposal if needed
proposal.segments[0].suggested_filename = "Custom_Name.pdf"

# Split the PDF
result = splitter.split_pdf(
    proposal=proposal,
    output_dir="/path/to/output"
)

# Check results
for segment in result.segments:
    print(f"Created: {segment.output_path}")
```

### 3. Session Manager (`session_manager.py`)

Provides stateful session management for web applications:

```python
from pdf_splitter.splitting import SplitSessionManager

# Initialize manager
manager = SplitSessionManager(db_path="/path/to/sessions.db")

# Create a session
session = manager.create_session(
    pdf_path="/path/to/document.pdf",
    proposal=proposal
)

# Update session status
manager.update_session(
    session_id=session.id,
    status=SessionStatus.CONFIRMED
)

# Track modifications
manager.update_session(
    session_id=session.id,
    modifications=[
        SessionModification(
            segment_index=0,
            field="suggested_filename",
            old_value="old_name.pdf",
            new_value="new_name.pdf"
        )
    ]
)

# Clean up expired sessions
manager.cleanup_expired_sessions()
```

## Features

### Intelligent Naming
- **Pattern Recognition**: Detects document types from content
- **Date Extraction**: Finds and uses most relevant dates
- **ID Extraction**: Captures invoice numbers, case numbers, etc.
- **Content Summarization**: Creates meaningful names from content

### Session Management
- **Persistence**: SQLite-based storage
- **State Tracking**: Full workflow from draft to complete
- **Modification History**: Tracks all user changes
- **Expiration**: Automatic cleanup of old sessions
- **Thread-Safe**: Safe for concurrent access

### Error Handling
- Graceful handling of corrupted PDFs
- Validation of page ranges
- Filesystem-safe filename generation
- Unique path collision resolution

## Testing

The module includes comprehensive tests:
```bash
# Run all tests
pytest pdf_splitter/splitting/tests/ -v

# Run with coverage
pytest pdf_splitter/splitting/tests/ --cov=pdf_splitter.splitting
```

Test coverage:
- Models: 100%
- PDFSplitter: 92%
- SessionManager: 93%

## Integration with Detection Module

The splitting module is designed to work seamlessly with the detection module:

```python
from pdf_splitter.detection import create_production_detector
from pdf_splitter.splitting import PDFSplitter

# Detect boundaries
detector = create_production_detector()
pages = detector.detect_boundaries(pdf_path)

# Generate split proposal
splitter = PDFSplitter()
proposal = splitter.generate_proposal(pdf_path, pages)

# Split the PDF
result = splitter.split_pdf(proposal, output_dir)
```

## Performance Considerations

- Uses pikepdf for efficient PDF operations
- Thread-safe for concurrent splitting
- Preview generation is optional (saves time)
- Session cleanup prevents database bloat
- Efficient page copying (no re-rendering)

## Future Enhancements

Potential improvements for future versions:
- OCR-based content extraction for better naming
- Machine learning for document type classification
- Batch processing optimization
- Cloud storage integration
- Advanced preview generation with thumbnails
