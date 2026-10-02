"""
report_generator.py
--------------------
Generates a PDF civic-issue report for a single detection record, using
reportlab. Consumes the same record shape that database.py stores/returns
so the GUI can call this straight off a history row or a fresh detection.
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)

REPORTS_DIR = Path(__file__).parent / "data" / "reports"


def _severity_color(severity):
    return {
        "High": colors.HexColor("#c0392b"),
        "Medium": colors.HexColor("#d68910"),
        "Low": colors.HexColor("#27ae60"),
    }.get(severity, colors.grey)


def generate_pdf_report(record, original_image_path=None, detected_image_path=None,
                        output_path=None):
    """
    record: dict with report fields.
    original_image_path / detected_image_path: image file paths to embed.
    output_path: where to save the PDF.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    original_image_path = original_image_path or record.get("image_path")
    detected_image_path = detected_image_path or record.get("detected_image_path")

    if output_path is None:
        rid = record.get("id", "new")
        output_path = str(REPORTS_DIR / f"report_{rid}.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "S2ATitle", parent=styles["Title"], textColor=colors.HexColor("#1a1a2e")
    )
    section_style = ParagraphStyle(
        "S2ASection", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    normal = styles["Normal"]

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story = []

    story.append(Paragraph("Snap2Action Civic Issue Report", title_style))
    created_at = record.get("created_at", "")
    story.append(Paragraph(f"Generated: {created_at}", normal))
    story.append(Spacer(1, 14))

    # Summary table
    issue_type = (record.get("issue_type") or "unknown").title()
    confidence = record.get("confidence", 0.0)
    severity = record.get("severity", "Unknown")
    mode = (record.get("mode") or "").title()
    bbox = record.get("bbox") or "N/A"
    area = record.get("area")
    area_str = f"{area} px\u00b2" if area else "N/A"

    summary_data = [
        ["Issue Type", issue_type],
        ["Surface Analyzed", mode or "N/A"],
        ["Confidence", f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)],
        ["Severity", severity],
        ["Affected Region (bbox)", str(bbox)],
        ["Affected Area", area_str],
        ["ROI Confidence", str(record.get("roi_confidence", "N/A"))],
    ]
    summary_table = Table(summary_data, colWidths=[2.2 * inch, 3.8 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef1f7")),
        ("TEXTCOLOR", (1, 3), (1, 3), _severity_color(severity)),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8c8d0")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # Embedded images
    for label, path in [("Original", original_image_path), ("Detected", detected_image_path)]:
        if path and Path(path).exists():
            story.append(Paragraph(label, section_style))
            try:
                img = Image(str(path), width=5.5 * inch, height=3.6 * inch, kind="proportional")
                story.append(img)
                story.append(Spacer(1, 10))
            except Exception:
                story.append(Paragraph(f"(could not embed image: {path})", normal))

    # Candidate breakdown
    candidates_raw = record.get("candidates_json") or record.get("candidates")
    candidates = []
    if isinstance(candidates_raw, str):
        try:
            candidates = json.loads(candidates_raw)
        except Exception:
            candidates = []
    elif isinstance(candidates_raw, list):
        candidates = candidates_raw

    if candidates:
        story.append(Paragraph("All Detected Candidates", section_style))
        cand_data = [["#", "Type", "Confidence", "BBox", "Area (px\u00b2)"]]
        for i, c in enumerate(candidates, start=1):
            cand_data.append([
                str(i),
                str(c.get("issue_type", "")).title(),
                f"{c.get('confidence', 0):.0%}" if isinstance(c.get("confidence"), float) else str(c.get("confidence")),
                str(c.get("bbox", "")),
                str(c.get("area", "")),
            ])
        cand_table = Table(cand_data, colWidths=[0.4 * inch, 1.1 * inch, 1.1 * inch, 1.8 * inch, 1.1 * inch])
        cand_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8c8d0")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")]),
        ]))
        story.append(cand_table)
        story.append(Spacer(1, 12))

    notes = record.get("notes")
    if notes:
        story.append(Paragraph("Notes", section_style))
        story.append(Paragraph(notes, normal))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report was generated automatically by Snap2Action_CV's OpenCV-based "
        "detection pipeline. Confidence and severity are algorithmic estimates and "
        "should be verified by a human inspector before action is taken.",
        ParagraphStyle("Disclaimer", parent=normal, textColor=colors.grey, fontSize=8),
    ))

    doc.build(story)
    return output_path
