#!/usr/bin/env python3
"""
Create a validation test set with different document patterns to test generalization.
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def create_validation_pdfs():
    """Create validation PDFs with known boundaries."""
    
    output_dir = Path("test_files/validation_set")
    output_dir.mkdir(exist_ok=True)
    
    styles = getSampleStyleSheet()
    
    # Test 1: Corporate documents (memos, reports, emails)
    print("Creating validation_corporate.pdf...")
    
    doc = SimpleDocTemplate(str(output_dir / "validation_corporate.pdf"), pagesize=letter)
    story = []
    boundaries = []
    page_num = 0
    
    # Document 1: Memo (2 pages)
    story.append(Paragraph("MEMORANDUM", styles['Title']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("TO: All Staff", styles['Normal']))
    story.append(Paragraph("FROM: Management", styles['Normal']))
    story.append(Paragraph("DATE: January 15, 2024", styles['Normal']))
    story.append(Paragraph("RE: New Policy Updates", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("This memo outlines important updates to our company policies...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("Continued from previous page...", styles['Normal']))
    story.append(Paragraph("In conclusion, these policy changes will take effect immediately.", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)  # Boundary before doc 2
    
    # Document 2: Email (1 page)
    story.append(Paragraph("From: john.doe@company.com", styles['Normal']))
    story.append(Paragraph("To: team@company.com", styles['Normal']))
    story.append(Paragraph("Subject: Project Update", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Hi Team,", styles['Normal']))
    story.append(Paragraph("Just wanted to give everyone a quick update on the project status...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Document 3: Report (3 pages)
    story.append(Paragraph("QUARTERLY SALES REPORT", styles['Title']))
    story.append(Paragraph("Q4 2023", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Executive Summary", styles['Heading3']))
    story.append(Paragraph("This quarter showed strong growth across all regions...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("Detailed Analysis", styles['Heading3']))
    story.append(Paragraph("Looking at the numbers more closely, we can see that...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("Recommendations", styles['Heading3']))
    story.append(Paragraph("Based on our analysis, we recommend the following actions...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Document 4: Short email (1 page)
    story.append(Paragraph("From: admin@company.com", styles['Normal']))
    story.append(Paragraph("To: all@company.com", styles['Normal']))
    story.append(Paragraph("Subject: Office Closed Tomorrow", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("The office will be closed tomorrow for maintenance.", styles['Normal']))
    
    doc.build(story)
    
    # Save ground truth
    with open(output_dir / "validation_corporate_truth.json", 'w') as f:
        json.dump({"boundaries": boundaries, "total_pages": page_num + 1}, f, indent=2)
    
    # Test 2: Academic papers style (with clear section breaks)
    print("Creating validation_academic.pdf...")
    
    doc = SimpleDocTemplate(str(output_dir / "validation_academic.pdf"), pagesize=letter)
    story = []
    boundaries = []
    page_num = 0
    
    # Paper 1 (3 pages)
    story.append(Paragraph("Machine Learning Applications in Healthcare", styles['Title']))
    story.append(Paragraph("John Smith, Jane Doe", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Abstract", styles['Heading2']))
    story.append(Paragraph("This paper explores the use of machine learning...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("1. Introduction", styles['Heading2']))
    story.append(Paragraph("Healthcare has seen significant advances...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("2. Methodology", styles['Heading2']))
    story.append(Paragraph("We employed a novel approach using...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Paper 2 (2 pages) - different style
    story.append(Paragraph("IMPACT OF CLIMATE CHANGE ON AGRICULTURE", styles['Title']))
    story.append(Paragraph("A. Johnson et al.", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("ABSTRACT", styles['Heading2']))
    story.append(Paragraph("Climate change poses significant challenges...", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("INTRODUCTION", styles['Heading2']))
    story.append(Paragraph("The agricultural sector is particularly vulnerable...", styles['Normal']))
    
    doc.build(story)
    
    with open(output_dir / "validation_academic_truth.json", 'w') as f:
        json.dump({"boundaries": boundaries, "total_pages": page_num + 1}, f, indent=2)
    
    # Test 3: Mixed with edge cases
    print("Creating validation_edge_cases.pdf...")
    
    doc = SimpleDocTemplate(str(output_dir / "validation_edge_cases.pdf"), pagesize=letter)
    story = []
    boundaries = []
    page_num = 0
    
    # Single page document
    story.append(Paragraph("NOTICE", styles['Title']))
    story.append(Paragraph("Building will be closed for renovation.", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Document starting with lowercase
    story.append(Paragraph("and furthermore, we believe that...", styles['Normal']))
    story.append(Paragraph("(This is actually a new document despite starting with lowercase)", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Very short pages
    story.append(Paragraph("Page 1", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    
    story.append(Paragraph("Page 2", styles['Normal']))
    story.append(PageBreak())
    page_num += 1
    boundaries.append(page_num - 1)
    
    # Normal document
    story.append(Paragraph("Meeting Minutes", styles['Title']))
    story.append(Paragraph("Date: March 1, 2024", styles['Normal']))
    story.append(Paragraph("Attendees: John, Jane, Bob", styles['Normal']))
    story.append(Paragraph("The meeting began at 2:00 PM...", styles['Normal']))
    
    doc.build(story)
    
    with open(output_dir / "validation_edge_cases_truth.json", 'w') as f:
        json.dump({"boundaries": boundaries, "total_pages": page_num + 1}, f, indent=2)
    
    print(f"\nCreated validation PDFs in {output_dir}")
    print("Files created:")
    print("  - validation_corporate.pdf (7 pages, 3 boundaries)")
    print("  - validation_academic.pdf (5 pages, 1 boundary)")
    print("  - validation_edge_cases.pdf (5 pages, 3 boundaries)")


if __name__ == "__main__":
    create_validation_pdfs()