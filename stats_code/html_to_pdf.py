#!/usr/bin/env python3
"""Render pipeline HTML reports to PDF (print media). Uses Playwright if installed."""

import os
from pathlib import Path


def _html_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def write_pdfs_from_html(output_dir: str) -> None:
    """Write comprehensive_analysis_report.pdf and manuscript_results_full_statistics.pdf next to HTML."""
    output_dir = os.path.abspath(output_dir)
    pairs = [
        ("comprehensive_analysis_report.html", "comprehensive_analysis_report.pdf"),
        ("manuscript_results_full_statistics.html", "manuscript_results_full_statistics.pdf"),
    ]
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "PDF generation skipped: install Playwright (pip install playwright && playwright install chromium)."
        )
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for html_name, pdf_name in pairs:
                html_path = os.path.join(output_dir, html_name)
                pdf_path = os.path.join(output_dir, pdf_name)
                if not os.path.isfile(html_path):
                    print(f"PDF skipped (missing): {html_path}")
                    continue
                page = browser.new_page()
                page.emulate_media(media="print")
                page.goto(_html_uri(html_path), wait_until="load", timeout=120000)
                page.pdf(
                    path=pdf_path,
                    format="A4",
                    print_background=True,
                    margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
                )
                page.close()
                print(f"Saved PDF: {pdf_path}")
        finally:
            browser.close()
