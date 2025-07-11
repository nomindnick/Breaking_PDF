#!/usr/bin/env python3
"""
Create a diverse test dataset for boundary detection.

This script generates synthetic PDFs with various document types and structures
to properly test the boundary detection module's generalization capabilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from datetime import datetime, timedelta
import random


def create_pdf_with_texts(output_path: Path, texts: List[str]):
    """Create a PDF with specified text content on each page."""
    doc = fitz.open()
    
    for text in texts:
        page = doc.new_page(width=612, height=792)  # Letter size
        # Add text to page with line wrapping
        y_position = 72  # Start 1 inch from top
        x_position = 72  # 1 inch from left
        line_height = 14  # Line spacing
        max_width = 468  # 6.5 inches
        
        # Split text into lines
        lines = text.split('\n')
        for line in lines:
            # Simple word wrapping
            words = line.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                # Estimate line width (rough approximation)
                if len(test_line) * 7 > max_width:  # ~7 pixels per character
                    if current_line:
                        page.insert_text((x_position, y_position), current_line, fontsize=12)
                        y_position += line_height
                        current_line = word
                    else:
                        # Word too long, insert anyway
                        page.insert_text((x_position, y_position), word, fontsize=12)
                        y_position += line_height
                        current_line = ""
                else:
                    current_line = test_line
            
            # Insert remaining text
            if current_line:
                page.insert_text((x_position, y_position), current_line, fontsize=12)
                y_position += line_height
            else:
                # Empty line
                y_position += line_height
            
            # Check if we're running out of space
            if y_position > 720:  # Leave 1 inch margin at bottom
                break
    
    doc.save(str(output_path))
    doc.close()


class DiverseTestDatasetGenerator:
    """Generate diverse test PDFs for boundary detection testing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_cases = []
    
    def generate_all_datasets(self):
        """Generate all test datasets."""
        print("Generating diverse test datasets...")
        
        # 1. Business documents
        self.create_business_documents()
        
        # 2. Legal documents
        self.create_legal_documents()
        
        # 3. Technical documents
        self.create_technical_documents()
        
        # 4. Mixed formats
        self.create_mixed_format_documents()
        
        # 5. Edge cases
        self.create_edge_cases()
        
        # 6. Real-world scenarios
        self.create_real_world_scenarios()
        
        # Save ground truth
        self.save_ground_truth()
        
        print(f"Generated {len(self.test_cases)} test cases")
    
    def create_business_documents(self):
        """Create typical business document scenarios."""
        # Test case 1: Invoices
        pages = []
        boundaries = set()
        
        # Invoice 1 (pages 0-1)
        pages.append(
            "INVOICE\n\n"
            "Invoice #: INV-2024-001\n"
            "Date: January 15, 2024\n\n"
            "Bill To:\n"
            "Acme Corporation\n"
            "123 Main Street\n"
            "Anytown, ST 12345\n\n"
            "Description                     Amount\n"
            "-" * 40 + "\n"
            "Consulting Services            $5,000.00\n"
            "Software License               $2,500.00"
        )
        pages.append(
            "Invoice #: INV-2024-001 (continued)\n\n"
            "Additional Services            $1,500.00\n"
            "-" * 40 + "\n"
            "Subtotal:                      $9,000.00\n"
            "Tax (8%):                        $720.00\n"
            "Total Due:                     $9,720.00\n\n"
            "Payment Terms: Net 30\n"
            "Thank you for your business!"
        )
        
        # Invoice 2 (pages 2-3)
        boundaries.add(1)
        pages.append(
            "INVOICE\n\n"
            "Invoice #: INV-2024-002\n"
            "Date: January 20, 2024\n\n"
            "Bill To:\n"
            "Beta Industries\n"
            "456 Oak Avenue\n"
            "Another City, ST 67890\n\n"
            "Description                     Amount\n"
            "-" * 40 + "\n"
            "Product A (100 units)          $3,000.00\n"
            "Product B (50 units)           $1,250.00\n"
            "Shipping                         $150.00"
        )
        pages.append(
            "Invoice #: INV-2024-002 (continued)\n\n"
            "-" * 40 + "\n"
            "Subtotal:                      $4,400.00\n"
            "Tax (8%):                        $352.00\n"
            "Total Due:                     $4,752.00\n\n"
            "Payment Terms: Due on receipt\n"
            "Please remit payment to the address above."
        )
        
        # Purchase order (pages 4-5)
        boundaries.add(3)
        pages.append(
            "PURCHASE ORDER\n\n"
            "PO Number: PO-2024-0156\n"
            "Date: January 22, 2024\n"
            "Vendor: Supply Co LLC\n\n"
            "Ship To:\n"
            "Warehouse #3\n"
            "789 Industrial Blvd\n"
            "Commerce City, ST 11111\n\n"
            "Item #    Description         Qty    Unit Price\n"
            "-" * 50
        )
        pages.append(
            "PO-2024-0156 (Page 2)\n\n"
            "A001      Widget Type A       500    $2.50\n"
            "B002      Gadget Type B       200    $5.00\n"
            "C003      Component C         1000   $0.75\n\n"
            "Total Order Value: $3,000.00\n\n"
            "Delivery Date: February 1, 2024\n"
            "Terms: Net 60"
        )
        
        pdf_path = self.output_dir / "test_business_documents.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Business Documents",
            "description": "Invoices and purchase orders",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["invoice", "invoice", "purchase_order"],
            "difficulty": "easy"
        })
    
    def create_legal_documents(self):
        """Create legal document scenarios."""
        pages = []
        boundaries = set()
        
        # Contract (pages 0-2)
        pages.append(
            "SERVICE AGREEMENT\n\n"
            "This Service Agreement (\"Agreement\") is entered into as of "
            "January 1, 2024 (\"Effective Date\") by and between:\n\n"
            "Client Corp (\"Client\")\n"
            "123 Business Way\n"
            "Corporate City, ST 12345\n\n"
            "AND\n\n"
            "Service Provider Inc (\"Provider\")\n"
            "456 Service Street\n"
            "Provider Town, ST 67890\n\n"
            "WHEREAS, Client desires to engage Provider for certain services;\n"
            "WHEREAS, Provider agrees to provide such services;\n\n"
            "NOW, THEREFORE, in consideration of the mutual covenants and "
            "agreements herein contained, the parties agree as follows:"
        )
        pages.append(
            "SERVICE AGREEMENT (Page 2)\n\n"
            "1. SERVICES\n"
            "Provider shall provide the following services (\"Services\"):\n"
            "- Software development and maintenance\n"
            "- Technical support and consultation\n"
            "- System integration services\n\n"
            "2. TERM\n"
            "This Agreement shall commence on the Effective Date and continue "
            "for a period of one (1) year, unless terminated earlier.\n\n"
            "3. COMPENSATION\n"
            "Client shall pay Provider a monthly fee of $10,000 for the Services."
        )
        pages.append(
            "SERVICE AGREEMENT (Page 3)\n\n"
            "4. CONFIDENTIALITY\n"
            "Each party agrees to maintain the confidentiality of the other "
            "party's proprietary information.\n\n"
            "5. TERMINATION\n"
            "Either party may terminate this Agreement with 30 days written notice.\n\n"
            "IN WITNESS WHEREOF, the parties have executed this Agreement.\n\n"
            "CLIENT CORP                    SERVICE PROVIDER INC\n\n"
            "_____________________         _____________________\n"
            "Authorized Signature           Authorized Signature\n\n"
            "Date: _______________         Date: _______________"
        )
        
        # Non-disclosure agreement (pages 3-4)
        boundaries.add(2)
        pages.append(
            "NON-DISCLOSURE AGREEMENT\n\n"
            "This Non-Disclosure Agreement (\"NDA\") is made as of January 15, 2024\n\n"
            "BETWEEN:\n"
            "Disclosing Party: Innovation Labs LLC\n"
            "Receiving Party: Partner Corp\n\n"
            "1. DEFINITION OF CONFIDENTIAL INFORMATION\n"
            "\"Confidential Information\" means any and all information disclosed "
            "by Disclosing Party to Receiving Party, including but not limited to:\n"
            "- Technical data, trade secrets, know-how\n"
            "- Research, products, services\n"
            "- Business plans and strategies"
        )
        pages.append(
            "NDA (Page 2)\n\n"
            "2. OBLIGATIONS OF RECEIVING PARTY\n"
            "Receiving Party agrees to:\n"
            "a) Hold Confidential Information in strict confidence\n"
            "b) Not disclose to third parties without written consent\n"
            "c) Use Confidential Information solely for evaluation purposes\n\n"
            "3. TERM\n"
            "This NDA shall remain in effect for five (5) years.\n\n"
            "AGREED AND ACCEPTED:\n\n"
            "Innovation Labs LLC            Partner Corp\n"
            "_____________________         _____________________"
        )
        
        pdf_path = self.output_dir / "test_legal_documents.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Legal Documents",
            "description": "Contracts and NDAs",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["contract", "nda"],
            "difficulty": "medium"
        })
    
    def create_technical_documents(self):
        """Create technical document scenarios."""
        pages = []
        boundaries = set()
        
        # Technical report (pages 0-2)
        pages.append(
            "TECHNICAL REPORT\n\n"
            "System Performance Analysis\n"
            "Date: January 25, 2024\n"
            "Author: Engineering Team\n\n"
            "EXECUTIVE SUMMARY\n\n"
            "This report presents the findings from our comprehensive system "
            "performance analysis conducted during Q4 2023. Key findings include:\n\n"
            "- Overall system uptime: 99.7%\n"
            "- Average response time: 125ms\n"
            "- Peak load capacity: 10,000 concurrent users"
        )
        pages.append(
            "Technical Report (Page 2)\n\n"
            "1. METHODOLOGY\n\n"
            "Performance testing was conducted using industry-standard tools:\n"
            "- Load testing: Apache JMeter\n"
            "- Monitoring: Prometheus and Grafana\n"
            "- Analysis period: October 1 - December 31, 2023\n\n"
            "2. RESULTS\n\n"
            "2.1 Response Time Analysis\n"
            "Average: 125ms\n"
            "95th percentile: 250ms\n"
            "99th percentile: 500ms"
        )
        pages.append(
            "Technical Report (Page 3)\n\n"
            "3. RECOMMENDATIONS\n\n"
            "Based on our analysis, we recommend:\n"
            "1. Upgrade database servers to handle increased load\n"
            "2. Implement caching layer for frequently accessed data\n"
            "3. Optimize query performance for slow endpoints\n\n"
            "4. CONCLUSION\n\n"
            "The system performs well under current load but requires "
            "optimization for future growth."
        )
        
        # Bug report (pages 3-4)
        boundaries.add(2)
        pages.append(
            "BUG REPORT\n\n"
            "Bug ID: BUG-2024-0789\n"
            "Date: January 26, 2024\n"
            "Reporter: QA Team\n"
            "Severity: High\n"
            "Status: Open\n\n"
            "SUMMARY:\n"
            "Login page crashes when special characters are entered in password field\n\n"
            "STEPS TO REPRODUCE:\n"
            "1. Navigate to login page\n"
            "2. Enter valid username\n"
            "3. Enter password containing: <>&\"\n"
            "4. Click login button"
        )
        pages.append(
            "Bug Report BUG-2024-0789 (continued)\n\n"
            "EXPECTED RESULT:\n"
            "System should either accept special characters or show validation error\n\n"
            "ACTUAL RESULT:\n"
            "Page shows 500 Internal Server Error\n\n"
            "ENVIRONMENT:\n"
            "- Browser: Chrome 120.0\n"
            "- OS: Windows 11\n"
            "- Server: Production\n\n"
            "NOTES:\n"
            "Issue appears to be related to improper input sanitization"
        )
        
        pdf_path = self.output_dir / "test_technical_documents.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Technical Documents",
            "description": "Technical reports and bug reports",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["technical_report", "bug_report"],
            "difficulty": "medium"
        })
    
    def create_mixed_format_documents(self):
        """Create documents with mixed formats and layouts."""
        pages = []
        boundaries = set()
        
        # Form (pages 0-1)
        pages.append(
            "APPLICATION FORM\n\n"
            "Personal Information\n"
            "-" * 40 + "\n"
            "Name: ________________________\n"
            "Date of Birth: _______________\n"
            "SSN: XXX-XX-____\n"
            "Address: _____________________\n"
            "         _____________________\n"
            "City: _________ State: __ ZIP: _____\n"
            "Phone: ______________________\n"
            "Email: ______________________"
        )
        pages.append(
            "APPLICATION FORM (Page 2)\n\n"
            "Employment History\n"
            "-" * 40 + "\n"
            "Current Employer: _____________\n"
            "Position: ____________________\n"
            "Start Date: __________________\n"
            "Salary: ______________________\n\n"
            "Previous Employer: ____________\n"
            "Position: ____________________\n"
            "Dates: __________ to _________\n\n"
            "Signature: ___________________\n"
            "Date: ________________________"
        )
        
        # Table/spreadsheet data (pages 2-3)
        boundaries.add(1)
        pages.append(
            "SALES REPORT - Q4 2023\n\n"
            "Month     | Product A | Product B | Product C | Total\n"
            "-" * 55 + "\n"
            "October   | $45,000   | $32,000   | $28,000   | $105,000\n"
            "November  | $52,000   | $38,000   | $31,000   | $121,000\n"
            "December  | $68,000   | $45,000   | $42,000   | $155,000\n"
            "-" * 55 + "\n"
            "Q4 Total  | $165,000  | $115,000  | $101,000  | $381,000\n\n"
            "Year-over-year growth: 15.2%"
        )
        pages.append(
            "SALES REPORT - Regional Breakdown\n\n"
            "Region      | Q3 2023   | Q4 2023   | Change\n"
            "-" * 45 + "\n"
            "Northeast   | $95,000   | $110,000  | +15.8%\n"
            "Southeast   | $78,000   | $85,000   | +9.0%\n"
            "Midwest     | $62,000   | $71,000   | +14.5%\n"
            "West        | $105,000  | $115,000  | +9.5%\n"
            "-" * 45 + "\n"
            "Total       | $340,000  | $381,000  | +12.1%"
        )
        
        # Presentation/slides (pages 4-5)
        boundaries.add(3)
        pages.append(
            "PROJECT KICKOFF\n"
            "━" * 40 + "\n\n"
            "                AGENDA\n\n"
            "    • Project Overview\n"
            "    • Timeline & Milestones\n"
            "    • Team Assignments\n"
            "    • Budget Review\n"
            "    • Q&A Session\n\n\n"
            "            January 2024"
        )
        pages.append(
            "KEY MILESTONES\n"
            "━" * 40 + "\n\n"
            "    Phase 1: Planning (Jan-Feb)\n"
            "      ✓ Requirements gathering\n"
            "      ✓ Design documentation\n\n"
            "    Phase 2: Development (Mar-Jun)\n"
            "      ✓ Core features\n"
            "      ✓ Testing\n\n"
            "    Phase 3: Deployment (Jul)\n"
            "      ✓ Production release\n"
            "      ✓ Training"
        )
        
        pdf_path = self.output_dir / "test_mixed_formats.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Mixed Format Documents",
            "description": "Forms, tables, and presentations",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["form", "spreadsheet", "presentation"],
            "difficulty": "hard"
        })
    
    def create_edge_cases(self):
        """Create edge case scenarios."""
        # Edge case 1: Very short documents
        pages = []
        boundaries = set()
        
        pages.append("Invoice #123")
        boundaries.add(0)
        pages.append("Receipt")
        boundaries.add(1)
        pages.append("")  # Empty page
        boundaries.add(2)
        pages.append("END")
        
        pdf_path = self.output_dir / "test_edge_very_short.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Very Short Documents",
            "description": "Single line and empty pages",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["invoice", "receipt", "empty", "note"],
            "difficulty": "hard"
        })
        
        # Edge case 2: No clear boundaries
        pages = []
        boundaries = set()
        
        pages.append(
            "Notes from meeting\n\n"
            "Discussed project timeline\n"
            "Budget needs review\n"
            "Next steps unclear"
        )
        pages.append(
            "More notes\n\n"
            "Team assignments pending\n"
            "Resource allocation TBD\n"
            "Follow up next week"
        )
        pages.append(
            "Additional thoughts\n\n"
            "Risk assessment needed\n"
            "Stakeholder buy-in required\n"
            "Documentation incomplete"
        )
        
        pdf_path = self.output_dir / "test_edge_no_boundaries.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "No Clear Boundaries",
            "description": "Continuous notes without clear document breaks",
            "file": pdf_path.name,
            "boundaries": [],  # No boundaries - all one document
            "document_types": ["notes"],
            "difficulty": "hard"
        })
        
        # Edge case 3: Repeated headers
        pages = []
        boundaries = set()
        
        for i in range(6):
            if i % 3 == 0 and i > 0:
                boundaries.add(i-1)
            pages.append(
                f"DAILY REPORT\n"
                f"Date: January {20+i//3}, 2024\n\n"
                f"Page {i%3 + 1} of 3\n\n"
                f"Section {i%3 + 1} content here..."
            )
        
        pdf_path = self.output_dir / "test_edge_repeated_headers.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Repeated Headers",
            "description": "Multiple documents with identical header patterns",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["report", "report"],
            "difficulty": "hard"
        })
    
    def create_real_world_scenarios(self):
        """Create realistic multi-document scenarios."""
        pages = []
        boundaries = set()
        
        # Email thread (pages 0-3)
        pages.append(
            "From: john.doe@company.com\n"
            "To: team@company.com\n"
            "Date: Mon, Jan 22, 2024 at 9:15 AM\n"
            "Subject: Project Update\n\n"
            "Team,\n\n"
            "Quick update on the project status:\n"
            "- Development is on track\n"
            "- Testing phase begins next week\n"
            "- Budget is within limits\n\n"
            "Please review and let me know if you have questions.\n\n"
            "Best,\nJohn"
        )
        pages.append(
            "From: jane.smith@company.com\n"
            "To: team@company.com\n"
            "Date: Mon, Jan 22, 2024 at 10:30 AM\n"
            "Subject: Re: Project Update\n\n"
            "Thanks for the update, John.\n\n"
            "Can we schedule a meeting to discuss the testing plan?\n"
            "I have some concerns about the timeline.\n\n"
            "Also, who will be leading the QA efforts?\n\n"
            "Jane"
        )
        pages.append(
            "From: mike.wilson@company.com\n"
            "To: team@company.com\n"
            "Date: Mon, Jan 22, 2024 at 11:45 AM\n"
            "Subject: Re: Project Update\n\n"
            "Hi all,\n\n"
            "I can lead the QA efforts. I've already started preparing\n"
            "the test cases and environments.\n\n"
            "Meeting tomorrow at 2 PM works for me.\n\n"
            "Mike"
        )
        pages.append(
            "From: john.doe@company.com\n"
            "To: team@company.com\n"
            "Date: Mon, Jan 22, 2024 at 1:00 PM\n"
            "Subject: Re: Project Update - Meeting Scheduled\n\n"
            "Perfect. Meeting scheduled for tomorrow at 2 PM.\n"
            "I'll send the calendar invite shortly.\n\n"
            "Agenda:\n"
            "1. Testing timeline review\n"
            "2. QA team assignments\n"
            "3. Risk assessment\n\n"
            "See you all there.\nJohn"
        )
        
        # Meeting minutes (pages 4-5)
        boundaries.add(3)
        pages.append(
            "MEETING MINUTES\n\n"
            "Date: January 23, 2024\n"
            "Time: 2:00 PM - 3:00 PM\n"
            "Subject: Project Testing Planning\n\n"
            "Attendees:\n"
            "- John Doe (Project Manager)\n"
            "- Jane Smith (Developer)\n"
            "- Mike Wilson (QA Lead)\n"
            "- Sarah Johnson (Business Analyst)\n\n"
            "AGENDA ITEMS:\n\n"
            "1. Testing Timeline Review\n"
            "   - Start date: January 29\n"
            "   - End date: February 16\n"
            "   - 3-week testing cycle approved"
        )
        pages.append(
            "Meeting Minutes (continued)\n\n"
            "2. QA Team Assignments\n"
            "   - Mike Wilson: Lead QA, test automation\n"
            "   - Tom Brown: Manual testing, regression\n"
            "   - Lisa Davis: Performance testing\n\n"
            "3. Risk Assessment\n"
            "   - Limited testing time: Mitigate with automation\n"
            "   - Resource constraints: Approved contractor support\n\n"
            "ACTION ITEMS:\n"
            "- Mike: Finalize test plan by Jan 25\n"
            "- Jane: Complete dev fixes by Jan 28\n"
            "- John: Secure additional resources\n\n"
            "Next meeting: January 30, 2024"
        )
        
        # Status report (pages 6-7)
        boundaries.add(5)
        pages.append(
            "WEEKLY STATUS REPORT\n\n"
            "Week Ending: January 26, 2024\n"
            "Project: Customer Portal v2.0\n\n"
            "ACCOMPLISHMENTS THIS WEEK:\n"
            "• Completed user authentication module\n"
            "• Fixed 15 bugs from previous sprint\n"
            "• Deployed to staging environment\n"
            "• Conducted security review\n\n"
            "PLANNED FOR NEXT WEEK:\n"
            "• Begin integration testing\n"
            "• Complete API documentation\n"
            "• Performance optimization"
        )
        pages.append(
            "Status Report (Page 2)\n\n"
            "ISSUES AND RISKS:\n"
            "• Database performance concerns\n"
            "  - Impact: Medium\n"
            "  - Mitigation: Index optimization planned\n\n"
            "• Third-party API delays\n"
            "  - Impact: Low\n"
            "  - Mitigation: Built fallback mechanism\n\n"
            "METRICS:\n"
            "• Sprint velocity: 45 points (target: 40)\n"
            "• Bug count: 23 open (down from 38)\n"
            "• Test coverage: 78% (target: 80%)\n\n"
            "Submitted by: John Doe\n"
            "Date: January 26, 2024"
        )
        
        pdf_path = self.output_dir / "test_real_world_scenario.pdf"
        create_pdf_with_texts(pdf_path, pages)
        
        self.test_cases.append({
            "name": "Real World Scenario",
            "description": "Email thread, meeting minutes, and status report",
            "file": pdf_path.name,
            "boundaries": sorted(list(boundaries)),
            "document_types": ["email_thread", "meeting_minutes", "status_report"],
            "difficulty": "medium"
        })
    
    def save_ground_truth(self):
        """Save ground truth data for all test cases."""
        ground_truth = {
            "description": "Diverse test dataset for boundary detection",
            "created": datetime.now().isoformat(),
            "test_cases": self.test_cases
        }
        
        output_path = self.output_dir / "diverse_test_ground_truth.json"
        with open(output_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Ground truth saved to: {output_path}")
        
        # Also create a summary
        summary_path = self.output_dir / "test_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("DIVERSE TEST DATASET SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            for tc in self.test_cases:
                f.write(f"Test Case: {tc['name']}\n")
                f.write(f"Description: {tc['description']}\n")
                f.write(f"Difficulty: {tc['difficulty']}\n")
                f.write(f"Documents: {len(tc['document_types'])}\n")
                f.write(f"Boundaries: {tc['boundaries']}\n")
                f.write("\n")
            
            f.write(f"\nTotal test cases: {len(self.test_cases)}\n")
            f.write(f"Total boundaries: {sum(len(tc['boundaries']) for tc in self.test_cases)}\n")
        
        print(f"Summary saved to: {summary_path}")


def main():
    """Generate diverse test dataset."""
    output_dir = Path("test_files/diverse_tests")
    generator = DiverseTestDatasetGenerator(output_dir)
    generator.generate_all_datasets()


if __name__ == "__main__":
    main()