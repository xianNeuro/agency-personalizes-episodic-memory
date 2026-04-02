#!/usr/bin/env python3
"""Render pipeline HTML reports to PDF (print media). Uses Playwright if installed."""

import os
from pathlib import Path


def _html_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def write_pdfs_from_html(output_dir: str) -> None:
    """Write comprehensive_analysis_report.pdf and manuscript_results_full_statistics.pdf next to HTML.

    Manuscript SI PDF uses US Letter (612×792 pt) to match Nature Communications supplement exports
    (e.g. Li_et_al_Supplement.pdf). Comprehensive technical appendix stays A4.
    """
    output_dir = os.path.abspath(output_dir)
    # (html, pdf, playwright format name)
    pairs = [
        ("comprehensive_analysis_report.html", "comprehensive_analysis_report.pdf", "A4"),
        (
            "manuscript_results_full_statistics.html",
            "manuscript_results_full_statistics.pdf",
            "Letter",
        ),
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
            for html_name, pdf_name, page_format in pairs:
                html_path = os.path.join(output_dir, html_name)
                pdf_path = os.path.join(output_dir, pdf_name)
                if not os.path.isfile(html_path):
                    print(f"PDF skipped (missing): {html_path}")
                    continue
                page = browser.new_page()
                page.emulate_media(media="print")
                page.goto(_html_uri(html_path), wait_until="load", timeout=120000)
                # ~1 in margins on Letter matches many journal SI PDFs (8.5"×11" page)
                margin_mm = "25.4mm" if page_format == "Letter" else "20mm"
                margin = {
                    "top": margin_mm,
                    "bottom": margin_mm,
                    "left": margin_mm,
                    "right": margin_mm,
                }
                page.pdf(
                    path=pdf_path,
                    format=page_format,
                    print_background=True,
                    margin=margin,
                )
                page.close()
                print(f"Saved PDF ({page_format}): {pdf_path}")
        finally:
            browser.close()
