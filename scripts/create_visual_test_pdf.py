#!/usr/bin/env python3
"""
Create a comprehensive test PDF for visual boundary detection experiments.

This script generates a diverse PDF with multiple document types to test
visual boundary detection techniques.
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, gray
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Document tracking for ground truth
documents_created = []
current_page = 1


def track_document(
    doc_type: str, pages: int, description: str, visual_markers: Dict[str, Any]
):
    """Track document for ground truth generation."""
    global current_page
    documents_created.append(
        {
            "type": doc_type,
            "start_page": current_page,
            "end_page": current_page + pages - 1,
            "pages": f"{current_page}-{current_page + pages - 1}",
            "description": description,
            "visual_markers": visual_markers,
        }
    )
    current_page += pages


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds page numbers."""

    def __init__(self, *args, **kwargs):
        """Initialize numbered canvas."""
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.page_offset = 0

    def showPage(self):
        """Show page and save state."""
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Save document with page numbers."""
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number()
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self):
        """Draw page number on current page."""
        self.setFont("Helvetica", 9)
        self.setFillColor(gray)
        self.drawRightString(
            letter[0] - 0.5 * inch,
            0.5 * inch,
            f"Page {self._pageNumber + self.page_offset}",
        )


def create_business_letter(
    story: List, styles: dict, company_name: str, letter_num: int
) -> int:
    """Create a business letter with letterhead."""
    # Custom letterhead style
    letterhead_style = ParagraphStyle(
        "Letterhead",
        parent=styles["Title"],
        fontSize=18,
        textColor=HexColor("#003366"),
        spaceAfter=6,
    )

    # Company letterhead
    story.append(Paragraph(f"<b>{company_name}</b>", letterhead_style))
    story.append(
        Paragraph(
            "123 Business Ave, Suite 456<br/>New York, NY 10001<br/>Tel: (555) 123-4567",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.5 * inch))

    # Date
    date = datetime.now() - timedelta(days=random.randint(1, 30))
    story.append(Paragraph(date.strftime("%B %d, %Y"), styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Recipient
    recipients = [
        ("John Smith", "ABC Corporation", "456 Main St", "Boston, MA 02101"),
        ("Jane Doe", "XYZ Industries", "789 Park Ave", "Chicago, IL 60601"),
        (
            "Robert Johnson",
            "Tech Solutions Inc",
            "321 Tech Blvd",
            "San Francisco, CA 94105",
        ),
    ]
    recipient = recipients[letter_num % len(recipients)]

    story.append(
        Paragraph(
            f"{recipient[0]}<br/>{recipient[1]}<br/>{recipient[2]}<br/>{recipient[3]}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Salutation
    story.append(Paragraph(f"Dear {recipient[0].split()[0]}:", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Body paragraphs
    body_options = [
        "I am writing to follow up on our recent discussion regarding the proposed partnership between our companies. As we discussed, I believe there are significant synergies that could benefit both organizations.",
        "Thank you for your interest in our products and services. We are pleased to provide you with the requested information and pricing details as outlined in the attached proposal.",
        "Following our meeting last week, I wanted to summarize the key points we discussed and outline the next steps for moving forward with this project.",
    ]

    story.append(
        Paragraph(body_options[letter_num % len(body_options)], styles["JustifiedBody"])
    )
    story.append(Spacer(1, 0.2 * inch))

    # Add 1-2 more paragraphs
    more_content = [
        "Our team has extensive experience in delivering similar solutions, and we are confident that we can meet your requirements within the specified timeline and budget. We have successfully completed over 50 similar projects in the past three years.",
        "The proposed solution includes comprehensive support and maintenance services, ensuring smooth operation and minimal downtime. Our 24/7 support team is always available to address any concerns.",
        "We understand the importance of maintaining the highest standards of quality and security. Our processes are ISO certified and we maintain strict compliance with all relevant regulations.",
    ]

    story.append(Paragraph(random.choice(more_content), styles["JustifiedBody"]))
    story.append(Spacer(1, 0.2 * inch))

    # Closing
    story.append(
        Paragraph(
            "Please feel free to contact me if you have any questions or need additional information. I look forward to hearing from you soon.",
            styles["JustifiedBody"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Signature
    story.append(Paragraph("Sincerely,", styles["Normal"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            "Michael Anderson<br/>Vice President of Sales<br/>manderson@company.com",
            styles["Normal"],
        )
    )

    # Page break
    story.append(PageBreak())

    pages = 2 if random.random() > 0.5 else 1
    if pages == 2:
        # Add second page with additional content
        story.append(Paragraph("Additional Information", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            Paragraph(
                "Please find below the detailed specifications and pricing information as requested:",
                styles["JustifiedBody"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Add a simple table
        data = [
            ["Service", "Description", "Price"],
            ["Basic Plan", "Essential features for small teams", "$99/month"],
            ["Professional", "Advanced features and priority support", "$299/month"],
            ["Enterprise", "Custom solutions with dedicated support", "Contact us"],
        ]
        t = Table(data, colWidths=[1.5 * inch, 3 * inch, 1.5 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(t)
        story.append(PageBreak())

    track_document(
        "Business Letter",
        pages,
        f"Letter from {company_name}",
        {
            "has_letterhead": True,
            "letterhead_color": "#003366",
            "consistent_margins": True,
            "has_signature_block": True,
        },
    )

    return pages


def create_technical_document(story: List, styles: dict, doc_num: int) -> int:
    """Create a technical document with code and diagrams."""
    # Title
    story.append(
        Paragraph(f"Technical Specification Document #{doc_num + 1}", styles["Title"])
    )
    story.append(Paragraph("API Integration Guide", styles["Heading1"]))
    story.append(Spacer(1, 0.3 * inch))

    # Introduction
    story.append(Paragraph("1. Introduction", styles["Heading2"]))
    story.append(
        Paragraph(
            "This document provides comprehensive technical specifications for integrating with our REST API. "
            "All endpoints follow RESTful conventions and return JSON responses.",
            styles["JustifiedBody"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Code section
    story.append(Paragraph("2. Authentication", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Code block style
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontSize=9,
        leftIndent=20,
        rightIndent=20,
        backColor=HexColor("#f5f5f5"),
        borderColor=HexColor("#cccccc"),
        borderWidth=1,
        borderPadding=10,
    )

    code_example = """curl -X POST https://api.example.com/auth/token \\
  -H "Content-Type: application/json" \\
  -d '{
    "client_id": "your_client_id",
    "client_secret": "your_secret",
    "grant_type": "client_credentials"
  }'"""

    story.append(Paragraph(code_example.replace("\n", "<br/>"), code_style))
    story.append(Spacer(1, 0.2 * inch))

    # Add a table with endpoints
    story.append(Paragraph("3. Available Endpoints", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    endpoints_data = [
        ["Method", "Endpoint", "Description"],
        ["GET", "/api/v1/users", "List all users"],
        ["POST", "/api/v1/users", "Create new user"],
        ["GET", "/api/v1/users/{id}", "Get user details"],
        ["PUT", "/api/v1/users/{id}", "Update user"],
        ["DELETE", "/api/v1/users/{id}", "Delete user"],
    ]

    t = Table(endpoints_data, colWidths=[1 * inch, 2.5 * inch, 2.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, HexColor("#E7E6E6")],
                ),
            ]
        )
    )
    story.append(t)

    story.append(PageBreak())

    # Second page with response examples
    story.append(Paragraph("4. Response Examples", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    json_example = """{
  "status": "success",
  "data": {
    "id": 12345,
    "username": "johndoe",
    "email": "john@example.com",
    "created_at": "2024-01-15T10:30:00Z"
  }
}"""

    story.append(Paragraph("Successful Response:", styles["Normal"]))
    story.append(Paragraph(json_example.replace("\n", "<br/>"), code_style))
    story.append(Spacer(1, 0.2 * inch))

    # Error handling
    story.append(Paragraph("5. Error Handling", styles["Heading2"]))
    story.append(
        Paragraph(
            "All error responses follow a consistent format with appropriate HTTP status codes. "
            "Common error codes include 400 (Bad Request), 401 (Unauthorized), 404 (Not Found), and 500 (Internal Server Error).",
            styles["JustifiedBody"],
        )
    )

    story.append(PageBreak())

    track_document(
        "Technical Document",
        2,
        f"API Technical Specification #{doc_num + 1}",
        {
            "has_code_blocks": True,
            "has_tables": True,
            "monospace_font": True,
            "technical_formatting": True,
        },
    )

    return 2


def create_invoice(story: List, styles: dict, invoice_num: int) -> int:
    """Create an invoice with structured data."""
    # Invoice header
    invoice_style = ParagraphStyle(
        "InvoiceHeader",
        parent=styles["Title"],
        fontSize=24,
        textColor=HexColor("#1a1a1a"),
        alignment=TA_RIGHT,
    )

    story.append(Paragraph("INVOICE", invoice_style))
    story.append(Spacer(1, 0.2 * inch))

    # Invoice details table
    invoice_date = datetime.now() - timedelta(days=random.randint(1, 60))
    due_date = invoice_date + timedelta(days=30)

    header_data = [
        ["", ""],
        ["Invoice #:", f"INV-{1000 + invoice_num}"],
        ["Date:", invoice_date.strftime("%B %d, %Y")],
        ["Due Date:", due_date.strftime("%B %d, %Y")],
    ]

    header_table = Table(header_data, colWidths=[4.5 * inch, 2 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 0.3 * inch))

    # Billing information
    billing_data = [
        ["BILL TO:", "INVOICE FROM:"],
        [
            "Customer Corp.\n456 Client Street\nClient City, ST 12345\nPhone: (555) 987-6543",
            "Service Provider Inc.\n789 Provider Ave\nProvider City, ST 67890\nPhone: (555) 123-4567",
        ],
    ]

    billing_table = Table(billing_data, colWidths=[3 * inch, 3 * inch])
    billing_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    story.append(billing_table)
    story.append(Spacer(1, 0.4 * inch))

    # Line items
    items_data = [
        ["Description", "Quantity", "Rate", "Amount"],
        ["Professional Services - Consulting", "40 hrs", "$150.00", "$6,000.00"],
        ["Software Development", "80 hrs", "$175.00", "$14,000.00"],
        ["Project Management", "20 hrs", "$125.00", "$2,500.00"],
        ["Technical Documentation", "10 hrs", "$100.00", "$1,000.00"],
    ]

    # Add more random items
    if random.random() > 0.5:
        items_data.extend(
            [
                ["Quality Assurance Testing", "30 hrs", "$120.00", "$3,600.00"],
                ["Database Design", "15 hrs", "$160.00", "$2,400.00"],
            ]
        )

    items_table = Table(
        items_data, colWidths=[3.5 * inch, 1 * inch, 1 * inch, 1 * inch]
    )
    items_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, HexColor("#f9f9f9")],
                ),
            ]
        )
    )
    story.append(items_table)
    story.append(Spacer(1, 0.3 * inch))

    # Totals
    subtotal = sum(
        float(item[3].replace("$", "").replace(",", "")) for item in items_data[1:]
    )
    tax = subtotal * 0.08
    total = subtotal + tax

    totals_data = [
        ["", "Subtotal:", f"${subtotal:,.2f}"],
        ["", "Tax (8%):", f"${tax:,.2f}"],
        ["", "TOTAL:", f"${total:,.2f}"],
    ]

    totals_table = Table(totals_data, colWidths=[4 * inch, 1.5 * inch, 1 * inch])
    totals_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("FONTNAME", (1, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LINEABOVE", (1, -1), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(totals_table)

    # Payment terms
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Payment Terms:", styles["Heading3"]))
    story.append(
        Paragraph(
            "Payment is due within 30 days. Please include invoice number with payment.",
            styles["Normal"],
        )
    )

    story.append(PageBreak())

    track_document(
        "Invoice",
        1,
        f"Invoice INV-{1000 + invoice_num}",
        {
            "has_tables": True,
            "structured_data": True,
            "numeric_content": True,
            "grid_layout": True,
        },
    )

    return 1


def create_email_thread(story: List, styles: dict, thread_num: int) -> int:
    """Create an email thread with multiple messages."""
    # Email header style
    email_header_style = ParagraphStyle(
        "EmailHeader",
        parent=styles["Normal"],
        fontSize=11,
        textColor=HexColor("#333333"),
        backColor=HexColor("#f0f0f0"),
        borderColor=HexColor("#cccccc"),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=12,
    )

    email_body_style = ParagraphStyle(
        "EmailBody",
        parent=styles["JustifiedBody"],
        fontSize=11,
        leftIndent=10,
        spaceAfter=8,
    )

    # Email thread subjects
    subjects = [
        "Project Update - Q4 Deliverables",
        "Meeting Schedule for Next Week",
        "Budget Approval Request",
        "New Feature Requirements",
    ]

    subject = subjects[thread_num % len(subjects)]
    story.append(Paragraph(f"<b>Email Thread: {subject}</b>", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    # Generate 3-5 emails in the thread
    num_emails = random.randint(3, 5)
    participants = [
        ("Sarah Johnson", "sjohnson@company.com"),
        ("David Chen", "dchen@company.com"),
        ("Maria Garcia", "mgarcia@company.com"),
        ("James Wilson", "jwilson@company.com"),
    ]

    for i in range(num_emails):
        # Email header
        sender = participants[i % len(participants)]
        timestamp = datetime.now() - timedelta(hours=i * 4 + random.randint(1, 8))

        header_text = f"""<b>From:</b> {sender[0]} &lt;{sender[1]}&gt;<br/>
<b>To:</b> team@company.com<br/>
<b>Date:</b> {timestamp.strftime('%B %d, %Y at %I:%M %p')}<br/>
<b>Subject:</b> {'Re: ' * i}{subject}"""

        story.append(Paragraph(header_text, email_header_style))

        # Email body
        if i == 0:
            body = f"""Hi Team,

I wanted to provide an update on our current progress and outline the next steps for the project.

Key accomplishments this week:
• Completed initial design mockups
• Finalized technical requirements
• Set up development environment
• Conducted stakeholder interviews

Please review the attached documents and let me know if you have any questions or concerns.

Best regards,
{sender[0].split()[0]}"""
        else:
            responses = [
                f"Thanks for the update, {participants[0][0].split()[0]}. The progress looks great. I have a few questions about the timeline. Can we discuss this in tomorrow's meeting?",
                "I've reviewed the documents. Everything looks good from my end. One suggestion: we might want to consider adding more buffer time for testing.",
                "Great work on the stakeholder interviews. The feedback will be invaluable. I'll have my team review the technical requirements by EOD.",
                f"Adding to {participants[(i-1) % len(participants)][0].split()[0]}'s point - we should also consider the integration timeline with the existing systems.",
            ]
            body = responses[i % len(responses)] + f"\n\nBest,\n{sender[0].split()[0]}"

        story.append(Paragraph(body.replace("\n", "<br/>"), email_body_style))
        story.append(Spacer(1, 0.3 * inch))

        # Add quoted text for replies (gray and indented)
        if i > 0:
            quoted_style = ParagraphStyle(
                "Quoted",
                parent=email_body_style,
                fontSize=10,
                textColor=HexColor("#666666"),
                leftIndent=20,
                borderLeftWidth=2,
                borderLeftColor=HexColor("#cccccc"),
                borderPadding=5,
            )
            story.append(Paragraph("<i>--- Original Message ---</i>", quoted_style))
            story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())

    pages = 2 if num_emails > 3 else 1
    if pages == 2:
        # Continue thread on second page
        story.append(Paragraph("(Email thread continued...)", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Add final summary email
        final_sender = participants[-1]
        story.append(
            Paragraph(
                f"<b>From:</b> {final_sender[0]} &lt;{final_sender[1]}&gt;<br/>"
                f"<b>To:</b> team@company.com<br/>"
                f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
                f"<b>Subject:</b> {'Re: ' * num_emails}{subject} - Summary",
                email_header_style,
            )
        )

        story.append(
            Paragraph(
                "Team,<br/><br/>"
                "Thanks for the productive discussion. Here's a summary of action items:<br/><br/>"
                "1. Design team to incorporate feedback by Friday<br/>"
                "2. Development to start implementation next Monday<br/>"
                "3. QA to prepare test plans by end of week<br/>"
                "4. Follow-up meeting scheduled for next Tuesday<br/><br/>"
                "Let me know if I missed anything.<br/><br/>"
                f"Thanks,<br/>{final_sender[0].split()[0]}",
                email_body_style,
            )
        )

        story.append(PageBreak())

    track_document(
        "Email Thread",
        pages,
        f"Email thread: {subject}",
        {
            "has_quoted_text": True,
            "indented_replies": True,
            "timestamp_headers": True,
            "gray_quoted_sections": True,
        },
    )

    return pages


def create_form(story: List, styles: dict, form_num: int) -> int:
    """Create a form document with fields and checkboxes."""
    form_types = [
        ("Employment Application", "Please complete all sections of this application"),
        ("Customer Feedback Survey", "Your opinion matters to us"),
        ("Service Request Form", "Complete this form to request services"),
        ("Registration Form", "Register for our upcoming event"),
    ]

    form_type = form_types[form_num % len(form_types)]

    # Form header
    form_header_style = ParagraphStyle(
        "FormHeader",
        parent=styles["Title"],
        fontSize=18,
        textColor=HexColor("#000080"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    story.append(Paragraph(form_type[0], form_header_style))
    story.append(Paragraph(form_type[1], styles["Italic"]))
    story.append(Spacer(1, 0.3 * inch))

    # Personal Information section
    story.append(Paragraph("Section 1: Personal Information", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Create form fields table
    field_data = [
        ["First Name: _______________________", "Last Name: _______________________"],
        ["Email: _____________________________", "Phone: ___________________________"],
        ["Address: ____________________________________________________________"],
        ["City: ____________________", "State: _____", "ZIP: __________"],
    ]

    field_table = Table(field_data, colWidths=[3 * inch, 3 * inch])
    field_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ]
        )
    )
    story.append(field_table)
    story.append(Spacer(1, 0.3 * inch))

    # Checkboxes section
    if form_num % 2 == 0:
        story.append(Paragraph("Section 2: Areas of Interest", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        checkbox_items = [
            "☐ Product Updates",
            "☐ Technical Support",
            "☐ Training Programs",
            "☐ Special Offers",
            "☐ Industry News",
            "☐ Webinars and Events",
        ]

        for item in checkbox_items:
            story.append(Paragraph(item, styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))
    else:
        story.append(Paragraph("Section 2: Service Selection", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        service_data = [
            ["☐ Basic Package", "$99/month", "☐ Premium Package", "$299/month"],
            ["☐ Professional", "$199/month", "☐ Enterprise", "Contact us"],
        ]

        service_table = Table(
            service_data, colWidths=[1.5 * inch, 1 * inch, 1.5 * inch, 1 * inch]
        )
        service_table.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        story.append(service_table)

    story.append(Spacer(1, 0.3 * inch))

    # Text area section
    story.append(Paragraph("Section 3: Additional Comments", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Create lined area for comments
    for i in range(5):
        story.append(Paragraph("_" * 70, styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

    # Signature section
    story.append(Spacer(1, 0.3 * inch))
    sig_data = [
        ["Signature: _______________________________", "Date: ________________"]
    ]
    sig_table = Table(sig_data, colWidths=[4 * inch, 2 * inch])
    sig_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ]
        )
    )
    story.append(sig_table)

    story.append(PageBreak())

    track_document(
        "Form",
        1,
        form_type[0],
        {
            "has_form_fields": True,
            "has_checkboxes": True,
            "has_lines": True,
            "structured_layout": True,
        },
    )

    return 1


def create_report(story: List, styles: dict, report_num: int) -> int:
    """Create a multi-page report with headers/footers."""
    report_titles = [
        "Quarterly Business Review - Q4 2024",
        "Annual Performance Report",
        "Market Analysis Summary",
        "Project Status Report",
    ]

    title = report_titles[report_num % len(report_titles)]

    # Cover page
    story.append(Spacer(1, 2 * inch))
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=28,
        textColor=HexColor("#1a472a"),
        alignment=TA_CENTER,
        spaceAfter=24,
    )
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("Prepared by: Analytics Department", styles["Heading2"]))
    story.append(Paragraph(datetime.now().strftime("%B %Y"), styles["Heading3"]))
    story.append(PageBreak())

    # Table of Contents
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    toc_items = [
        ("1. Executive Summary", "3"),
        ("2. Key Findings", "4"),
        ("3. Detailed Analysis", "5"),
        ("4. Recommendations", "7"),
        ("5. Appendix", "8"),
    ]

    for item, page in toc_items:
        toc_line = f"{item}{'.' * (60 - len(item) - len(page))}{page}"
        story.append(Paragraph(toc_line, styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    exec_summary = """This report provides a comprehensive analysis of our performance over the reporting period.
    Key achievements include a 15% increase in revenue, successful launch of three new products, and expansion
    into two new markets. The following sections detail our findings and provide recommendations for continued growth."""

    story.append(Paragraph(exec_summary, styles["JustifiedBody"]))
    story.append(Spacer(1, 0.3 * inch))

    # Add bullet points
    story.append(Paragraph("Key Highlights:", styles["Heading3"]))
    highlights = [
        "Revenue growth exceeded projections by 5%",
        "Customer satisfaction scores improved to 92%",
        "Operational efficiency increased by 18%",
        "Market share expanded in all target segments",
    ]

    for highlight in highlights:
        story.append(Paragraph(f"• {highlight}", styles["Normal"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(PageBreak())

    # Key Findings with chart placeholder
    story.append(Paragraph("2. Key Findings", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    # Create a simple chart image
    chart_img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(chart_img)

    # Draw axes
    draw.line([(50, 250), (350, 250)], fill="black", width=2)  # X-axis
    draw.line([(50, 50), (50, 250)], fill="black", width=2)  # Y-axis

    # Draw bars
    bar_data = [120, 150, 180, 165, 190, 220]
    bar_width = 40
    x_start = 80

    for i, value in enumerate(bar_data):
        height = int(value * 0.8)
        draw.rectangle(
            [(x_start + i * 50, 250 - height), (x_start + i * 50 + bar_width, 250)],
            fill=tuple(int(c) for c in HexColor("#4472C4").rgb()),
            outline="black",
        )

    # Save and add to story
    chart_path = Path("/tmp/chart_temp.png")
    chart_img.save(chart_path)
    story.append(RLImage(str(chart_path), width=4 * inch, height=3 * inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "The above chart demonstrates consistent growth across all quarters.",
            styles["Normal"],
        )
    )

    # Continue with more pages...
    story.append(PageBreak())

    track_document(
        "Report",
        4,
        title,
        {
            "has_cover_page": True,
            "has_toc": True,
            "has_page_numbers": True,
            "has_charts": True,
            "multi_section": True,
        },
    )

    return 4


def create_mixed_content(story: List, styles: dict, doc_num: int) -> int:
    """Create a document with mixed content (text + images)."""
    titles = [
        "Product Catalog - Spring Collection",
        "Marketing Presentation",
        "Training Manual - Module 3",
        "Company Newsletter - December Edition",
    ]

    title = titles[doc_num % len(titles)]

    # Header
    header_style = ParagraphStyle(
        "MixedHeader",
        parent=styles["Title"],
        fontSize=20,
        textColor=HexColor("#ff6600"),
        spaceAfter=12,
    )

    story.append(Paragraph(title, header_style))
    story.append(Spacer(1, 0.2 * inch))

    # Introduction text
    intro = "Welcome to our latest edition featuring exciting updates and valuable information for our community."
    story.append(Paragraph(intro, styles["JustifiedBody"]))
    story.append(Spacer(1, 0.3 * inch))

    # Create image with text
    img = Image.new(
        "RGB", (500, 200), color=tuple(int(c) for c in HexColor("#e6f2ff").rgb())
    )
    draw = ImageDraw.Draw(img)

    # Add some shapes
    draw.rectangle(
        [(20, 20), (180, 180)], fill=tuple(int(c) for c in HexColor("#3366cc").rgb())
    )
    draw.ellipse(
        [(220, 40), (380, 160)], fill=tuple(int(c) for c in HexColor("#ff9900").rgb())
    )
    draw.polygon(
        [(420, 180), (480, 60), (460, 180)],
        fill=tuple(int(c) for c in HexColor("#109618").rgb()),
    )

    # Add text to image
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((200, 90), "Visual Content", fill="white", font=font)

    img_path = Path("/tmp/mixed_content_temp.png")
    img.save(img_path)
    story.append(RLImage(str(img_path), width=5 * inch, height=2 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # Two column layout
    left_text = """Feature Highlights:

• Enhanced user interface
• Improved performance
• New collaboration tools
• Mobile optimization
• Advanced analytics"""

    right_text = """Benefits:

• 50% faster processing
• Intuitive design
• Real-time updates
• Cross-platform support
• Detailed insights"""

    two_col_data = [
        [
            Paragraph(left_text.replace("\n", "<br/>"), styles["Normal"]),
            Paragraph(right_text.replace("\n", "<br/>"), styles["Normal"]),
        ]
    ]

    two_col_table = Table(two_col_data, colWidths=[3 * inch, 3 * inch])
    two_col_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(two_col_table)

    story.append(PageBreak())

    track_document(
        "Mixed Content",
        1,
        title,
        {
            "has_images": True,
            "has_shapes": True,
            "two_column_layout": True,
            "colorful_design": True,
        },
    )

    return 1


def create_presentation_slides(story: List, styles: dict, pres_num: int) -> int:
    """Create presentation-style pages."""
    presentations = [
        "Strategic Planning 2025",
        "Product Launch Overview",
        "Team Performance Review",
        "Market Expansion Strategy",
    ]

    title = presentations[pres_num % len(presentations)]

    # Slide 1 - Title slide
    slide_title_style = ParagraphStyle(
        "SlideTitle",
        parent=styles["Title"],
        fontSize=32,
        textColor=HexColor("#003366"),
        alignment=TA_CENTER,
        spaceBefore=2 * inch,
        spaceAfter=0.5 * inch,
    )

    slide_subtitle_style = ParagraphStyle(
        "SlideSubtitle",
        parent=styles["Heading2"],
        fontSize=18,
        textColor=HexColor("#666666"),
        alignment=TA_CENTER,
    )

    story.append(Paragraph(title, slide_title_style))
    story.append(Paragraph("Executive Presentation", slide_subtitle_style))
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), slide_subtitle_style))
    story.append(PageBreak())

    # Slide 2 - Bullet points
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Key Objectives", styles["Heading1"]))
    story.append(Spacer(1, 0.5 * inch))

    bullet_style = ParagraphStyle(
        "BulletPoint",
        parent=styles["Normal"],
        fontSize=16,
        leftIndent=40,
        spaceBefore=12,
        spaceAfter=12,
    )

    objectives = [
        "Increase market share by 25%",
        "Launch 3 new product lines",
        "Expand to 5 new regions",
        "Achieve 95% customer satisfaction",
    ]

    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))

    story.append(PageBreak())

    # Slide 3 - Simple diagram
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Growth Strategy", styles["Heading1"]))
    story.append(Spacer(1, 0.3 * inch))

    # Create a simple flow diagram
    diagram_img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(diagram_img)

    # Draw boxes
    boxes = [
        (50, 100, 150, 180, "Research"),
        (200, 100, 300, 180, "Develop"),
        (350, 100, 450, 180, "Test"),
        (500, 100, 600, 180, "Launch"),
    ]

    for x1, y1, x2, y2, text in boxes:
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            fill=tuple(int(c) for c in HexColor("#e6f2ff").rgb()),
            outline="black",
            width=2,
        )
        text_x = x1 + (x2 - x1) // 2 - len(text) * 3
        text_y = y1 + (y2 - y1) // 2 - 6
        draw.text((text_x, text_y), text, fill="black")

    # Draw arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][2]
        x2 = boxes[i + 1][0]
        y = 140
        draw.line([(x1, y), (x2, y)], fill="black", width=2)
        draw.polygon([(x2, y), (x2 - 5, y - 5), (x2 - 5, y + 5)], fill="black")

    diagram_path = Path("/tmp/diagram_temp.png")
    diagram_img.save(diagram_path)
    story.append(RLImage(str(diagram_path), width=5 * inch, height=2.5 * inch))

    story.append(PageBreak())

    track_document(
        "Presentation",
        3,
        title,
        {
            "slide_layout": True,
            "large_fonts": True,
            "centered_content": True,
            "minimal_text": True,
            "has_diagrams": True,
        },
    )

    return 3


def create_legal_document(story: List, styles: dict, doc_num: int) -> int:
    """Create a legal-style document."""
    doc_types = [
        ("SERVICE AGREEMENT", "This Service Agreement"),
        ("NON-DISCLOSURE AGREEMENT", "This Non-Disclosure Agreement"),
        ("TERMS AND CONDITIONS", "These Terms and Conditions"),
        ("SOFTWARE LICENSE AGREEMENT", "This Software License Agreement"),
    ]

    doc_type = doc_types[doc_num % len(doc_types)]

    # Legal header style
    legal_style = ParagraphStyle(
        "Legal",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=6,
    )

    # Title
    title_style = ParagraphStyle(
        "LegalTitle",
        parent=styles["Title"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=24,
    )

    story.append(Paragraph(doc_type[0], title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Parties
    story.append(
        Paragraph(
            f"{doc_type[1]} (\"Agreement\") is entered into as of {datetime.now().strftime('%B %d, %Y')} "
            f'("Effective Date") by and between Company ABC, a Delaware corporation ("Company") and '
            f'Client XYZ, a California corporation ("Client").',
            legal_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Recitals
    story.append(Paragraph("WHEREAS:", styles["Heading3"]))
    story.append(
        Paragraph(
            "A. Company desires to provide certain services to Client; and", legal_style
        )
    )
    story.append(
        Paragraph(
            "B. Client desires to obtain such services from Company.", legal_style
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "NOW, THEREFORE, in consideration of the mutual covenants and agreements hereinafter set forth "
            "and for other good and valuable consideration, the receipt and sufficiency of which are hereby "
            "acknowledged, the parties agree as follows:",
            legal_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Sections
    sections = [
        (
            "1. DEFINITIONS",
            "For purposes of this Agreement, the following terms shall have the meanings set forth below:",
        ),
        (
            "2. SERVICES",
            "Company shall provide to Client the services described in Exhibit A attached hereto.",
        ),
        (
            "3. TERM",
            "This Agreement shall commence on the Effective Date and continue for a period of one (1) year.",
        ),
        (
            "4. COMPENSATION",
            "In consideration for the Services, Client shall pay Company as set forth in Exhibit B.",
        ),
        (
            "5. CONFIDENTIALITY",
            "Each party acknowledges that it may have access to confidential information of the other party.",
        ),
    ]

    for section_title, section_text in sections:
        story.append(Paragraph(section_title, styles["Heading3"]))
        story.append(Paragraph(section_text, legal_style))
        story.append(Spacer(1, 0.15 * inch))

        # Add subsections for some sections
        if "SERVICES" in section_title:
            story.append(
                Paragraph(
                    "2.1 Company shall use commercially reasonable efforts to provide the Services in a professional and workmanlike manner.",
                    legal_style,
                )
            )
            story.append(
                Paragraph(
                    "2.2 Client shall provide Company with reasonable access to Client's facilities and personnel as necessary.",
                    legal_style,
                )
            )
            story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())

    # Signature page
    story.append(
        Paragraph(
            "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.",
            legal_style,
        )
    )
    story.append(Spacer(1, 0.5 * inch))

    sig_data = [
        ["COMPANY ABC:", "", "CLIENT XYZ:", ""],
        ["", "", "", ""],
        ["_______________________", "", "_______________________", ""],
        ["By: Name", "", "By: Name", ""],
        ["Title: CEO", "", "Title: CEO", ""],
        ["Date: _____________", "", "Date: _____________", ""],
    ]

    sig_table = Table(
        sig_data, colWidths=[1.5 * inch, 0.5 * inch, 1.5 * inch, 0.5 * inch]
    )
    sig_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(sig_table)

    story.append(PageBreak())

    track_document(
        "Legal Document",
        2,
        doc_type[0],
        {
            "justified_text": True,
            "numbered_sections": True,
            "formal_language": True,
            "signature_blocks": True,
        },
    )

    return 2


def create_test_pdf():
    """Create the comprehensive test PDF."""
    output_path = Path("test_files/visual_test_pdf.pdf")
    output_path.parent.mkdir(exist_ok=True)

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    # Build story
    story = []
    styles = getSampleStyleSheet()

    # Add custom styles
    if "JustifiedBody" not in styles:
        styles.add(
            ParagraphStyle(
                name="JustifiedBody",
                parent=styles["Normal"],
                fontSize=11,
                leading=14,
                alignment=TA_JUSTIFY,
            )
        )

    # Create various document types
    print("Creating test PDF with diverse document types...")

    # 1. Business Letters (3 different companies)
    for i in range(3):
        companies = [
            "Acme Corporation",
            "Global Tech Solutions",
            "Premier Services Inc.",
        ]
        create_business_letter(story, styles, companies[i], i)

    # 2. Technical Documents (2)
    for i in range(2):
        create_technical_document(story, styles, i)

    # 3. Invoices (3)
    for i in range(3):
        create_invoice(story, styles, i)

    # 4. Email Threads (3)
    for i in range(3):
        create_email_thread(story, styles, i)

    # 5. Forms (2)
    for i in range(2):
        create_form(story, styles, i)

    # 6. Reports (1 multi-page)
    create_report(story, styles, 0)

    # 7. Mixed Content (2)
    for i in range(2):
        create_mixed_content(story, styles, i)

    # 8. Presentations (2)
    for i in range(2):
        create_presentation_slides(story, styles, i)

    # 9. Legal Documents (2)
    for i in range(2):
        create_legal_document(story, styles, i)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)

    print(f"Test PDF created: {output_path}")
    print(f"Total pages: {current_page - 1}")
    print(f"Total documents: {len(documents_created)}")

    # Create ground truth JSON
    create_ground_truth()

    return output_path


def create_ground_truth():
    """Create ground truth JSON file."""
    ground_truth = {
        "pdf_info": {
            "filename": "visual_test_pdf.pdf",
            "total_pages": current_page - 1,
            "total_documents": len(documents_created),
            "creation_date": datetime.now().isoformat(),
        },
        "documents": [],
        "visual_boundaries": [],
    }

    # Add document information
    for doc in documents_created:
        ground_truth["documents"].append(
            {
                "pages": doc["pages"],
                "type": doc["type"],
                "description": doc["description"],
                "visual_markers": doc["visual_markers"],
            }
        )

        # Add boundary after each document (except the last)
        if doc != documents_created[-1]:
            ground_truth["visual_boundaries"].append(
                {
                    "after_page": doc["end_page"],
                    "confidence": 1.0,
                    "signals": list(doc["visual_markers"].keys()),
                    "boundary_type": "document_end",
                }
            )

    # Save ground truth
    output_path = Path("test_files/visual_test_pdf_ground_truth.json")
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Ground truth created: {output_path}")

    # Print summary
    print("\nDocument Summary:")
    for doc in documents_created:
        print(f"  {doc['type']}: pages {doc['pages']} - {doc['description']}")

    print(f"\nTotal boundaries: {len(ground_truth['visual_boundaries'])}")


if __name__ == "__main__":
    create_test_pdf()
