"""
Shared testing utilities for PDF Splitter tests.

This module provides helper functions and utilities that can be used
across different test modules.
"""

import io
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock

import fitz  # PyMuPDF
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pdf_splitter.preprocessing.pdf_handler import PageType

# --- PDF Generation Utilities ---


def create_test_pdf(
    num_pages: int = 5,
    page_size: Tuple[float, float] = (612, 792),  # Letter size
    include_text: bool = True,
    include_images: bool = False,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Create a test PDF with specified characteristics.

    Args:
        num_pages: Number of pages to create
        page_size: Page size as (width, height) in points
        include_text: Whether to include text content
        include_images: Whether to include images
        output_path: Path to save the PDF (optional)

    Returns:
        Path to created PDF if output_path provided, None otherwise
    """
    doc = fitz.open()

    for page_num in range(num_pages):
        page = doc.new_page(width=page_size[0], height=page_size[1])

        if include_text:
            # Add header
            header_text = f"Test Document - Page {page_num + 1}"
            page.insert_text(
                (50, 50),
                header_text,
                fontsize=16,
                fontname="helv",
                color=(0, 0, 0),
            )

            # Add body text
            body_text = f"This is the content of page {page_num + 1}. " * 10
            text_rect = fitz.Rect(50, 100, page_size[0] - 50, page_size[1] - 100)
            page.insert_textbox(
                text_rect,
                body_text,
                fontsize=12,
                fontname="helv",
                color=(0, 0, 0),
            )

            # Add footer
            footer_text = f"Page {page_num + 1} of {num_pages}"
            page.insert_text(
                (50, page_size[1] - 50),
                footer_text,
                fontsize=10,
                fontname="helv",
                color=(0.5, 0.5, 0.5),
            )

        if include_images:
            # Create a simple test image
            img = create_test_image_bytes(200, 150)
            img_rect = fitz.Rect(200, 300, 400, 450)
            page.insert_image(img_rect, stream=img)

    if output_path:
        doc.save(str(output_path))
        doc.close()
        return output_path
    else:
        doc.close()
        return None


def create_mixed_test_pdf(
    documents: List[Dict[str, Any]],
    output_path: Path,
    page_size: Tuple[float, float] = (612, 792),  # Letter size
) -> Path:
    """
    Create a comprehensive test PDF with mixed content types.

    Args:
        documents: List of document specifications with:
            - type: Document type (email, invoice, letter, rfi, etc.)
            - pages: Number of pages
            - page_type: "searchable", "scanned", or "mixed"
            - quality: For scanned pages - "high", "medium", or "low"
            - rotation: Rotation angle for scanned pages
            - add_noise: Whether to add noise to scanned pages
        output_path: Path to save the PDF
        page_size: Page size in points

    Returns:
        Path to created PDF
    """
    doc = fitz.open()
    page_counter = 0

    # Convert points to pixels for image generation (150 DPI for speed)
    dpi = 150
    img_width = int(page_size[0] * dpi / 72)
    img_height = int(page_size[1] * dpi / 72)
    img_size = (img_width, img_height)

    print(f"Creating PDF with {len(documents)} documents...")

    for doc_idx, doc_spec in enumerate(documents, 1):
        doc_type = doc_spec["type"]
        num_pages = doc_spec.get("pages", 1)
        page_type = doc_spec.get("page_type", "searchable")
        quality = doc_spec.get("quality", "high")
        rotation = doc_spec.get("rotation", 0.0)
        add_noise = doc_spec.get("add_noise", False)
        add_blur = doc_spec.get("add_blur", False)

        print(
            f"  Creating document {doc_idx}/{len(documents)}: {doc_type} "
            f"({num_pages} pages, {page_type})"
        )

        for page_num in range(num_pages):
            page_counter += 1

            # Create document content based on type
            if doc_type == "email":
                template = create_email_template(
                    page_num=page_num + 1,
                    total_pages=num_pages,
                )
            elif doc_type == "invoice":
                template = create_invoice_template(
                    invoice_no=f"INV-2024-{page_counter:03d}",
                    page_num=page_num + 1,
                    total_pages=num_pages,
                )
            elif doc_type == "letter":
                template = create_letter_template(
                    page_num=page_num + 1,
                    total_pages=num_pages,
                )
            elif doc_type == "rfi":
                template = create_rfi_template(
                    rfi_no=f"RFI-2024-{page_counter:03d}",
                    page_num=page_num + 1,
                    total_pages=num_pages,
                )
            else:
                # Generic document
                template = {
                    "type": "generic",
                    "title": f"{doc_type} - Page {page_num + 1}",
                    "content": generate_random_text(50, 200),
                    "page": f"Page {page_num + 1} of {num_pages}",
                }

            # Decide if this specific page should be scanned
            if page_type == "scanned" or (page_type == "mixed" and page_num % 2 == 1):
                # Create image-based page
                img_bytes = create_image_page(
                    content=template,
                    page_size=img_size,
                    quality=quality,
                    rotation=rotation,
                    add_noise=add_noise,
                    add_blur=add_blur,
                )

                # Add image page to PDF
                page = doc.new_page(width=page_size[0], height=page_size[1])
                img_rect = fitz.Rect(0, 0, page_size[0], page_size[1])
                page.insert_image(img_rect, stream=img_bytes)

            else:
                # Create searchable text page
                page = doc.new_page(width=page_size[0], height=page_size[1])

                # Render template as text
                if doc_type == "email":
                    _render_email_text(page, template, page_size)
                elif doc_type == "invoice":
                    _render_invoice_text(page, template, page_size)
                elif doc_type == "letter":
                    _render_letter_text(page, template, page_size)
                elif doc_type == "rfi":
                    _render_form_text(page, template, page_size)
                else:
                    # Generic text rendering
                    page.insert_text(
                        (50, 50),
                        template.get("title", "Document"),
                        fontsize=16,
                        fontname="helv",
                    )
                    page.insert_text(
                        (50, 100),
                        template.get("content", ""),
                        fontsize=12,
                        fontname="helv",
                    )

    doc.save(str(output_path))
    doc.close()
    return output_path


def _render_email_text(page, template: Dict[str, Any], page_size: Tuple[float, float]):
    """Render email template as searchable text."""
    y_pos = 50

    # Headers
    headers = [
        f"From: {template.get('from', '')}",
        f"To: {template.get('to', '')}",
        f"Date: {template.get('date', '')}",
        f"Subject: {template.get('subject', '')}",
    ]

    for header in headers:
        page.insert_text((50, y_pos), header, fontsize=11, fontname="helv")
        y_pos += 20

    # Separator
    y_pos += 10

    # Body
    body = template.get("body", "")
    for line in body.split("\n"):
        if y_pos < page_size[1] - 50:
            page.insert_text((50, y_pos), line, fontsize=12, fontname="helv")
            y_pos += 20


def _render_invoice_text(
    page, template: Dict[str, Any], page_size: Tuple[float, float]
):
    """Render invoice template as searchable text."""
    # Company name
    page.insert_text(
        (50, 50),
        template.get("company", ""),
        fontsize=16,
        fontname="helv",
    )

    # Invoice number and date
    page.insert_text(
        (50, 100),
        f"INVOICE #{template.get('invoice_no', '')}",
        fontsize=14,
        fontname="helv",
    )
    page.insert_text(
        (400, 100),
        f"Date: {template.get('date', '')}",
        fontsize=12,
        fontname="helv",
    )

    # Bill to
    page.insert_text((50, 150), "Bill To:", fontsize=12, fontname="helv")
    page.insert_text(
        (50, 170),
        template.get("bill_to", ""),
        fontsize=11,
        fontname="helv",
    )

    # Items
    y_pos = 250
    page.insert_text((50, y_pos), "Description", fontsize=12, fontname="helv")
    page.insert_text((450, y_pos), "Amount", fontsize=12, fontname="helv")

    y_pos += 30
    for item in template.get("items", []):
        page.insert_text((50, y_pos), item["desc"], fontsize=11, fontname="helv")
        page.insert_text((450, y_pos), item["amount"], fontsize=11, fontname="helv")
        y_pos += 20

    # Total
    y_pos += 20
    page.insert_text((400, y_pos), "Total:", fontsize=12, fontname="helv")
    page.insert_text(
        (450, y_pos), template.get("total", ""), fontsize=12, fontname="helv"
    )


def _render_letter_text(page, template: Dict[str, Any], page_size: Tuple[float, float]):
    """Render letter template as searchable text."""
    # Letterhead
    page.insert_text(
        (page_size[0] / 2 - 100, 50),
        template.get("letterhead", ""),
        fontsize=14,
        fontname="helv",
    )

    # Date
    page.insert_text((100, 120), template.get("date", ""), fontsize=12, fontname="helv")

    # Recipient
    y_pos = 170
    for line in template.get("recipient", "").split("\n"):
        page.insert_text((100, y_pos), line, fontsize=11, fontname="helv")
        y_pos += 18

    # Salutation
    y_pos += 20
    page.insert_text(
        (100, y_pos), template.get("salutation", ""), fontsize=12, fontname="helv"
    )

    # Body
    y_pos += 30
    body_rect = fitz.Rect(100, y_pos, page_size[0] - 100, page_size[1] - 150)
    page.insert_textbox(
        body_rect,
        template.get("body", ""),
        fontsize=11,
        fontname="helv",
    )

    # Closing
    page.insert_text(
        (100, page_size[1] - 120),
        template.get("closing", ""),
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        (100, page_size[1] - 60),
        template.get("signature", ""),
        fontsize=12,
        fontname="helv",
    )


def _render_form_text(page, template: Dict[str, Any], page_size: Tuple[float, float]):
    """Render form template as searchable text."""
    # Title
    page.insert_text(
        (page_size[0] / 2 - 150, 50),
        template.get("title", ""),
        fontsize=16,
        fontname="helv",
    )

    # Fields
    y_pos = 120
    for field in template.get("fields", []):
        page.insert_text((50, y_pos), field["label"], fontsize=11, fontname="helv")
        page.insert_text((200, y_pos), field["value"], fontsize=11, fontname="helv")
        y_pos += 25

    # Description
    y_pos += 20
    page.insert_text((50, y_pos), "Description:", fontsize=12, fontname="helv")
    y_pos += 30

    desc_rect = fitz.Rect(50, y_pos, page_size[0] - 50, y_pos + 200)
    page.insert_textbox(
        desc_rect,
        template.get("description", ""),
        fontsize=11,
        fontname="helv",
    )


def create_test_image_bytes(width: int = 200, height: int = 150) -> bytes:
    """Create a test image and return as bytes."""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw some shapes
    draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
    draw.text((20, 20), "Test Image", fill="black")

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def create_image_page(
    content: Union[str, Dict[str, Any]],
    page_size: Tuple[int, int] = (
        1240,
        1754,
    ),  # A4 at 150 DPI (reduced for faster generation)
    quality: str = "high",
    rotation: float = 0.0,
    add_noise: bool = False,
    add_blur: bool = False,
) -> bytes:
    """
    Create an image-based page simulating a scanned document.

    Args:
        content: Text content or document template dict
        page_size: Page size in pixels (width, height)
        quality: Scan quality - "high", "medium", "low"
        rotation: Rotation angle in degrees
        add_noise: Whether to add noise to simulate scan artifacts
        add_blur: Whether to add blur to simulate poor focus

    Returns:
        PNG image bytes
    """
    # Create base image
    if quality == "low":
        bg_color = (245, 245, 235)  # Slightly yellowed
        text_color = (60, 60, 60)  # Lighter black
    else:
        bg_color = (255, 255, 255)  # White
        text_color = (0, 0, 0)  # Black

    img = Image.new("RGB", page_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a better font, fall back to default if not available
    try:
        font_size = 40 if quality == "high" else 35 if quality == "medium" else 30
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size
        )
    except Exception:
        font = None  # Will use default font

    # Process content
    if isinstance(content, dict):
        # Render from template
        _render_document_template(draw, content, page_size, font, text_color)
    else:
        # Render plain text
        lines = content.split("\n")
        y_pos = 100
        x_margin = 100

        for line in lines:
            if font:
                draw.text((x_margin, y_pos), line, fill=text_color, font=font)
                y_pos += int(font_size * 1.5)
            else:
                draw.text((x_margin, y_pos), line, fill=text_color)
                y_pos += 25

    # Apply quality effects
    if quality == "medium":
        # Reduce resolution slightly
        small_size = (page_size[0] // 2, page_size[1] // 2)
        img = img.resize(small_size, Image.Resampling.BILINEAR)
        img = img.resize(page_size, Image.Resampling.BILINEAR)
    elif quality == "low":
        # Reduce resolution more
        small_size = (page_size[0] // 3, page_size[1] // 3)
        img = img.resize(small_size, Image.Resampling.NEAREST)
        img = img.resize(page_size, Image.Resampling.BILINEAR)

    # Apply rotation
    if rotation != 0:
        img = img.rotate(rotation, fillcolor=bg_color, expand=False)

    # Apply noise
    if add_noise:
        img_array = np.array(img)
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    # Apply blur
    if add_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG", optimize=True)
    return img_bytes.getvalue()


def _render_document_template(
    draw: ImageDraw.Draw,
    template: Dict[str, Any],
    page_size: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    text_color: Tuple[int, int, int],
):
    """Render a document template to an image."""
    doc_type = template.get("type", "generic")

    if doc_type == "email":
        _render_email_template(draw, template, page_size, font, text_color)
    elif doc_type == "invoice":
        _render_invoice_template(draw, template, page_size, font, text_color)
    elif doc_type == "letter":
        _render_letter_template(draw, template, page_size, font, text_color)
    elif doc_type == "form":
        _render_form_template(draw, template, page_size, font, text_color)
    else:
        # Generic rendering
        y_pos = 100
        for key, value in template.items():
            if key != "type":
                text = f"{key}: {value}" if key != "content" else str(value)
                draw.text((100, y_pos), text, fill=text_color, font=font)
                y_pos += 40


def _render_email_template(
    draw: ImageDraw.Draw,
    template: Dict[str, Any],
    page_size: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    text_color: Tuple[int, int, int],
):
    """Render an email document template."""
    x_margin = 100
    y_pos = 100
    line_height = 40 if font else 25

    # Email header
    headers = [
        f"From: {template.get('from', 'sender@example.com')}",
        f"To: {template.get('to', 'recipient@example.com')}",
        f"Date: {template.get('date', 'Mon, 15 Jan 2024 10:30:00 PST')}",
        f"Subject: {template.get('subject', 'Re: Project Update')}",
    ]

    for header in headers:
        draw.text((x_margin, y_pos), header, fill=text_color, font=font)
        y_pos += line_height

    # Separator line
    y_pos += 20
    draw.line(
        [(x_margin, y_pos), (page_size[0] - x_margin, y_pos)], fill=text_color, width=2
    )
    y_pos += 40

    # Email body
    body = template.get(
        "body", "This is the email body content.\n\nBest regards,\nSender"
    )
    for line in body.split("\n"):
        if y_pos < page_size[1] - 100:  # Leave margin at bottom
            draw.text((x_margin, y_pos), line, fill=text_color, font=font)
            y_pos += line_height


def _render_invoice_template(
    draw: ImageDraw.Draw,
    template: Dict[str, Any],
    page_size: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    text_color: Tuple[int, int, int],
):
    """Render an invoice document template."""
    x_margin = 100
    y_pos = 100
    line_height = 40 if font else 25

    # Company header
    draw.text(
        (x_margin, y_pos),
        template.get("company", "ACME Construction Inc."),
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 2

    # Invoice details
    draw.text(
        (x_margin, y_pos),
        f"INVOICE #{template.get('invoice_no', 'INV-2024-001')}",
        fill=text_color,
        font=font,
    )
    draw.text(
        (page_size[0] - 400, y_pos),
        f"Date: {template.get('date', '01/15/2024')}",
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 2

    # Bill to
    draw.text((x_margin, y_pos), "Bill To:", fill=text_color, font=font)
    y_pos += line_height
    draw.text(
        (x_margin + 50, y_pos),
        template.get("bill_to", "Client Company\n123 Main St\nCity, ST 12345"),
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 3

    # Table header
    draw.line(
        [(x_margin, y_pos), (page_size[0] - x_margin, y_pos)], fill=text_color, width=1
    )
    y_pos += 10
    draw.text((x_margin, y_pos), "Description", fill=text_color, font=font)
    draw.text((page_size[0] - 300, y_pos), "Amount", fill=text_color, font=font)
    y_pos += line_height
    draw.line(
        [(x_margin, y_pos), (page_size[0] - x_margin, y_pos)], fill=text_color, width=1
    )
    y_pos += 20

    # Line items
    items = template.get(
        "items",
        [
            {"desc": "Labor - 40 hours @ $150/hr", "amount": "$6,000.00"},
            {"desc": "Materials", "amount": "$2,500.00"},
        ],
    )

    for item in items:
        draw.text((x_margin, y_pos), item["desc"], fill=text_color, font=font)
        draw.text(
            (page_size[0] - 300, y_pos), item["amount"], fill=text_color, font=font
        )
        y_pos += line_height

    # Total
    y_pos += 20
    draw.line(
        [(page_size[0] - 400, y_pos), (page_size[0] - x_margin, y_pos)],
        fill=text_color,
        width=2,
    )
    y_pos += 20
    draw.text((page_size[0] - 400, y_pos), "Total:", fill=text_color, font=font)
    draw.text(
        (page_size[0] - 300, y_pos),
        template.get("total", "$8,500.00"),
        fill=text_color,
        font=font,
    )


def _render_letter_template(
    draw: ImageDraw.Draw,
    template: Dict[str, Any],
    page_size: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    text_color: Tuple[int, int, int],
):
    """Render a letter document template."""
    x_margin = 150
    y_pos = 150
    line_height = 40 if font else 25

    # Letterhead
    letterhead = template.get("letterhead", "PROFESSIONAL SERVICES LLC")
    draw.text((page_size[0] // 2 - 200, y_pos), letterhead, fill=text_color, font=font)
    y_pos += line_height * 2

    # Date
    draw.text(
        (x_margin, y_pos),
        template.get("date", "January 15, 2024"),
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 2

    # Recipient
    recipient = template.get(
        "recipient",
        "John Doe\nProject Manager\nConstruction Co.\n456 Oak Ave\nCity, ST 54321",
    )
    for line in recipient.split("\n"):
        draw.text((x_margin, y_pos), line, fill=text_color, font=font)
        y_pos += line_height

    y_pos += line_height

    # Salutation
    draw.text(
        (x_margin, y_pos),
        template.get("salutation", "Dear Mr. Doe:"),
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 2

    # Body
    body = template.get(
        "body",
        "I am writing to confirm our meeting scheduled for next week "
        "regarding the ongoing construction project.\n\n"
        "Please let me know if you need any additional information "
        "before our meeting.\n\n"
        "Thank you for your continued partnership.",
    )

    # Word wrap body text
    words = body.split()
    line = ""
    max_width = page_size[0] - (2 * x_margin)

    for word in words:
        test_line = line + " " + word if line else word
        # Approximate width check
        if len(test_line) * 20 > max_width:
            draw.text((x_margin, y_pos), line, fill=text_color, font=font)
            y_pos += line_height
            line = word
        else:
            line = test_line

    if line:
        draw.text((x_margin, y_pos), line, fill=text_color, font=font)
        y_pos += line_height

    # Closing
    y_pos += line_height
    draw.text(
        (x_margin, y_pos),
        template.get("closing", "Sincerely,"),
        fill=text_color,
        font=font,
    )
    y_pos += line_height * 3
    draw.text(
        (x_margin, y_pos),
        template.get("signature", "Jane Smith"),
        fill=text_color,
        font=font,
    )


def _render_form_template(
    draw: ImageDraw.Draw,
    template: Dict[str, Any],
    page_size: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    text_color: Tuple[int, int, int],
):
    """Render a form document template."""
    x_margin = 100
    y_pos = 100
    line_height = 50 if font else 30

    # Form title
    title = template.get("title", "REQUEST FOR INFORMATION")
    draw.text((page_size[0] // 2 - 300, y_pos), title, fill=text_color, font=font)
    y_pos += line_height * 2

    # Form fields
    fields = template.get(
        "fields",
        [
            {"label": "RFI Number:", "value": "RFI-2024-042"},
            {"label": "Date:", "value": "01/15/2024"},
            {"label": "Project:", "value": "Downtown Office Complex"},
            {"label": "To:", "value": "Design Team"},
            {"label": "From:", "value": "General Contractor"},
        ],
    )

    for field in fields:
        draw.text((x_margin, y_pos), field["label"], fill=text_color, font=font)
        draw.text((x_margin + 300, y_pos), field["value"], fill=text_color, font=font)
        # Draw field line
        draw.line(
            [(x_margin + 300, y_pos + 35), (x_margin + 700, y_pos + 35)],
            fill=text_color,
            width=1,
        )
        y_pos += line_height

    y_pos += line_height

    # Description section
    draw.text((x_margin, y_pos), "Description:", fill=text_color, font=font)
    y_pos += line_height

    # Draw box for description
    box_height = 300
    draw.rectangle(
        [x_margin, y_pos, page_size[0] - x_margin, y_pos + box_height],
        outline=text_color,
        width=2,
    )

    # Add description text
    desc_text = template.get(
        "description",
        "Please clarify the specification for the exterior cladding "
        "material on the north facade.",
    )
    draw.text((x_margin + 20, y_pos + 20), desc_text, fill=text_color, font=font)


# --- Document Template Creation Functions ---


def create_email_template(
    sender: str = "john.smith@construction.com",
    recipient: str = "jane.doe@client.com",
    subject: str = "RE: Project Status Update",
    date: Optional[datetime] = None,
    body: Optional[str] = None,
    page_num: int = 1,
    total_pages: int = 1,
) -> Dict[str, Any]:
    """Create an email document template."""
    if date is None:
        date = datetime.now()

    if body is None:
        body = (
            f"Hi Jane,\n\n"
            f"Thank you for your email regarding the project status. I wanted "
            f"to provide you with an update on our progress.\n\n"
            f"We have completed the foundation work and are currently working "
            f"on the structural framing. The project remains on schedule, and "
            f"we expect to complete this phase by the end of next week.\n\n"
            f"Please find attached the latest progress photos and updated "
            f"schedule.\n\n"
            f"Best regards,\n"
            f"John Smith\n"
            f"Project Manager\n\n"
            f"Page {page_num} of {total_pages}"
        )

    return {
        "type": "email",
        "from": sender,
        "to": recipient,
        "date": date.strftime("%a, %d %b %Y %H:%M:%S %Z"),
        "subject": subject,
        "body": body,
    }


def create_invoice_template(
    invoice_no: str = "INV-2024-001",
    company: str = "ABC Construction Services",
    bill_to: str = "XYZ Development Corp\n789 Business Blvd\nMetropolis, ST 98765",
    date: Optional[datetime] = None,
    items: Optional[List[Dict[str, str]]] = None,
    page_num: int = 1,
    total_pages: int = 1,
) -> Dict[str, Any]:
    """Create an invoice document template."""
    if date is None:
        date = datetime.now()

    if items is None:
        items = [
            {"desc": "Site Preparation - 80 hours @ $125/hr", "amount": "$10,000.00"},
            {"desc": "Concrete Work - 120 hours @ $150/hr", "amount": "$18,000.00"},
            {"desc": "Materials - Concrete, Rebar, Forms", "amount": "$15,750.00"},
            {"desc": "Equipment Rental - 2 weeks", "amount": "$3,500.00"},
        ]

    total = sum(
        float(item["amount"].replace("$", "").replace(",", "")) for item in items
    )

    return {
        "type": "invoice",
        "invoice_no": invoice_no,
        "company": company,
        "bill_to": bill_to,
        "date": date.strftime("%m/%d/%Y"),
        "items": items,
        "total": f"${total:,.2f}",
        "page_info": f"Page {page_num} of {total_pages}",
    }


def create_letter_template(
    letterhead: str = "SMITH & ASSOCIATES ENGINEERING",
    recipient: str = (
        "Ms. Sarah Johnson\nProject Director\n"
        "City Planning Department\n100 Civic Center Dr\nDowntown, ST 11111"
    ),
    subject: str = "Re: Permit Application - Downtown Office Complex",
    date: Optional[datetime] = None,
    body: Optional[str] = None,
    signature: str = "Robert Smith, P.E.",
    page_num: int = 1,
    total_pages: int = 1,
) -> Dict[str, Any]:
    """Create a letter document template."""
    if date is None:
        date = datetime.now()

    if body is None:
        body = (
            f"Following our conversation yesterday, I am writing to formally "
            f"submit our permit application for the Downtown Office Complex "
            f"project.\n\n"
            f"The application package includes all required documentation, "
            f"including:\n"
            f"- Completed permit application forms\n"
            f"- Architectural drawings and specifications\n"
            f"- Structural engineering calculations\n"
            f"- Environmental impact assessment\n"
            f"- Traffic study results\n\n"
            f"We have addressed all comments from the preliminary review and "
            f"believe the application is now complete. Please let me know if "
            f"you require any additional information.\n\n"
            f"We look forward to your review and approval of this application."
            f"\n\nPage {page_num} of {total_pages}"
        )

    return {
        "type": "letter",
        "letterhead": letterhead,
        "recipient": recipient,
        "date": date.strftime("%B %d, %Y"),
        "salutation": "Dear Ms. Johnson:",
        "body": body,
        "closing": "Respectfully submitted,",
        "signature": signature,
    }


def create_rfi_template(
    rfi_no: str = "RFI-2024-042",
    project: str = "Downtown Office Complex - Phase 2",
    to: str = "Architectural Team",
    from_: str = "General Contractor - BuildCo Inc.",
    date: Optional[datetime] = None,
    description: Optional[str] = None,
    page_num: int = 1,
    total_pages: int = 1,
) -> Dict[str, Any]:
    """Create an RFI (Request for Information) form template."""
    if date is None:
        date = datetime.now()

    if description is None:
        description = (
            f'The specifications call for "premium grade exterior cladding" '
            f"on the north facade, but the material schedule lists standard "
            f"grade aluminum panels.\n\n"
            f"Please clarify which material should be used and provide updated "
            f"specifications if necessary.\n\n"
            f"This RFI is critical for maintaining the project schedule as the "
            f"cladding installation is scheduled to begin next month.\n\n"
            f"Page {page_num} of {total_pages}"
        )

    return {
        "type": "form",
        "title": "REQUEST FOR INFORMATION (RFI)",
        "fields": [
            {"label": "RFI Number:", "value": rfi_no},
            {"label": "Date:", "value": date.strftime("%m/%d/%Y")},
            {"label": "Project:", "value": project},
            {"label": "To:", "value": to},
            {"label": "From:", "value": from_},
            {
                "label": "Response Required By:",
                "value": (date + timedelta(days=7)).strftime("%m/%d/%Y"),
            },
        ],
        "description": description,
    }


# --- Text Generation Utilities ---


def generate_random_text(
    min_words: int = 10,
    max_words: int = 100,
    seed: Optional[int] = None,
) -> str:
    """Generate random text for testing."""
    if seed:
        random.seed(seed)

    words = [
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "that",
        "is",
        "was",
        "for",
        "document",
        "page",
        "text",
        "content",
        "section",
        "paragraph",
        "title",
        "header",
        "footer",
        "table",
        "image",
        "figure",
        "data",
        "information",
        "analysis",
        "report",
        "summary",
    ]

    num_words = random.randint(min_words, max_words)
    text = " ".join(random.choices(words, k=num_words))

    # Capitalize first letter and add period
    return text[0].upper() + text[1:] + "."


def create_document_segments(num_segments: int = 3) -> List[Tuple[int, int]]:
    """Create document segment boundaries for testing."""
    segments = []
    start = 0

    for i in range(num_segments):
        length = random.randint(2, 10)
        end = start + length - 1
        segments.append((start, end))
        start = end + 1

    return segments


# --- Mock Creation Utilities ---


def create_mock_pdf_page(
    text: str = "Sample text",
    page_num: int = 0,
    width: float = 612,
    height: float = 792,
) -> Mock:
    """Create a mock PDF page with specified properties."""
    page = Mock()
    page.number = page_num
    page.rect = Mock(width=width, height=height)
    page.get_text.return_value = text

    # Mock text extraction with blocks
    blocks = []
    lines = text.split("\n")
    y_pos = 50

    for i, line in enumerate(lines):
        if line.strip():
            blocks.append(
                {
                    "type": 0,  # Text block
                    "bbox": (50, y_pos, 500, y_pos + 20),
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": line,
                                    "font": "Arial",
                                    "size": 12,
                                    "flags": 0,
                                }
                            ]
                        }
                    ],
                }
            )
            y_pos += 30

    page.get_text.return_value = {"blocks": blocks}

    # Mock pixmap for rendering
    pixmap = Mock()
    pixmap.width = int(width)
    pixmap.height = int(height)
    pixmap.samples = b"\xff" * (pixmap.width * pixmap.height * 3)
    page.get_pixmap.return_value = pixmap

    return page


def create_mock_ocr_result(
    text: str = "OCR extracted text",
    confidence: float = 0.95,
    page_num: int = 0,
) -> Dict[str, Any]:
    """Create a mock OCR result."""
    lines = text.split("\n")
    text_lines = []

    for i, line in enumerate(lines):
        if line.strip():
            text_lines.append(
                {
                    "text": line,
                    "confidence": confidence + random.uniform(-0.05, 0.05),
                    "bbox": {
                        "x1": 50,
                        "y1": 50 + i * 30,
                        "x2": 500,
                        "y2": 70 + i * 30,
                    },
                }
            )

    return {
        "page_num": page_num,
        "text_lines": text_lines,
        "full_text": text,
        "avg_confidence": confidence,
        "processing_time": random.uniform(0.5, 2.0),
        "word_count": len(text.split()),
        "char_count": len(text),
    }


# --- Assertion Helpers ---


def assert_pdf_valid(pdf_path: Path):
    """Assert that a PDF file is valid and can be opened."""
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"

    try:
        doc = fitz.open(str(pdf_path))
        assert doc.page_count > 0, "PDF has no pages"
        doc.close()
    except Exception as e:
        pytest.fail(f"Failed to open PDF: {e}")


def assert_text_quality(
    text: str,
    min_length: int = 10,
    min_words: int = 2,
    max_error_ratio: float = 0.1,
):
    """Assert text meets quality requirements."""
    assert len(text) >= min_length, f"Text too short: {len(text)} < {min_length}"

    words = text.split()
    assert len(words) >= min_words, f"Too few words: {len(words)} < {min_words}"

    # Check for common OCR errors
    error_chars = sum(1 for c in text if c in "�□▯")
    error_ratio = error_chars / len(text) if text else 0
    assert (
        error_ratio <= max_error_ratio
    ), f"Too many error characters: {error_ratio:.2%}"


def assert_ocr_result_valid(result: Dict[str, Any]):
    """Assert OCR result has valid structure and content."""
    required_fields = [
        "page_num",
        "text_lines",
        "full_text",
        "avg_confidence",
        "processing_time",
        "word_count",
        "char_count",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    assert isinstance(result["text_lines"], list)
    assert 0 <= result["avg_confidence"] <= 1
    assert result["processing_time"] >= 0
    assert result["word_count"] >= 0
    assert result["char_count"] >= 0


# --- Performance Testing Helpers ---


def measure_performance(func, *args, iterations: int = 3, **kwargs) -> Dict[str, float]:
    """Measure function performance over multiple iterations."""
    import time

    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "total": sum(times),
        "iterations": iterations,
    }


def assert_performance(
    elapsed_time: float,
    max_time: float,
    operation: str = "Operation",
):
    """Assert that performance meets requirements."""
    assert (
        elapsed_time <= max_time
    ), f"{operation} took too long: {elapsed_time:.2f}s > {max_time:.2f}s"


# --- Image Testing Utilities ---


def create_noisy_image(
    width: int = 400,
    height: int = 200,
    noise_level: float = 0.1,
) -> np.ndarray:
    """Create a noisy test image."""
    # Start with white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add text-like black rectangles
    for i in range(3):
        y_start = 50 + i * 50
        y_end = y_start + 20
        x_end = width - 50 - i * 50
        img[y_start:y_end, 50:x_end] = 0

    # Add noise
    noise = np.random.random((height, width, 3))
    noise_mask = noise < noise_level / 2
    img[noise_mask] = 0  # Salt noise
    noise_mask = noise > (1 - noise_level / 2)
    img[noise_mask] = 255  # Pepper noise

    return img


def compare_images(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compare two images and return similarity score (0-1)."""
    if img1.shape != img2.shape:
        return 0.0

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    # Calculate normalized difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = 255.0
    similarity = 1.0 - (np.mean(diff) / max_diff)

    return similarity


# --- Data Validation Helpers ---


def validate_page_type(page_type: PageType) -> bool:
    """Validate that page type is valid enum value."""
    return page_type in [
        PageType.SEARCHABLE,
        PageType.IMAGE_BASED,
        PageType.MIXED,
        PageType.EMPTY,
    ]


def validate_confidence_score(score: float) -> bool:
    """Validate confidence score is in valid range."""
    return 0.0 <= score <= 1.0


def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate bounding box has valid coordinates."""
    x1, y1, x2, y2 = bbox
    return x1 <= x2 and y1 <= y2 and all(v >= 0 for v in bbox)
