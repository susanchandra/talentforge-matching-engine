"""
src/parser.py — Resume PDF and text parsing for TalentForge.

Supports PDF files (via PyMuPDF) and raw text strings.
"""

import os
from typing import Union


def parse_resume(source: Union[str, bytes]) -> str:
    """
    Parse a resume from a PDF file path or raw text string.

    Args:
        source: Either a file path ending in '.pdf' or a raw resume text string.

    Returns:
        Extracted resume text as a single string.

    Raises:
        ValueError: If extracted text is under 100 characters (insufficient content).
        FileNotFoundError: If the provided PDF path does not exist.
    """
    text = ""

    # Determine if source is a PDF file path
    if isinstance(source, str) and source.strip().lower().endswith(".pdf"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF parsing. Install with: pip install PyMuPDF")

        doc = fitz.open(source)
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            pages_text.append(page_text)
        doc.close()

        text = "\n".join(pages_text)

    elif isinstance(source, str):
        # Treat as raw text
        text = source

    else:
        raise TypeError(f"Unsupported source type: {type(source)}. Expected str (text or PDF path).")

    # Normalize whitespace while preserving structure
    text = text.strip()

    # Validation: ensure meaningful content was extracted
    if len(text) < 100:
        raise ValueError(
            f"Resume parsing produced insufficient text. "
            f"Got {len(text)} characters (minimum 100 required). "
            f"Check that the PDF is not scanned/image-only or that text was provided."
        )

    return text
