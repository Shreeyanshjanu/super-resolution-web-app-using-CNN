#!/usr/bin/env python3
"""
Generate a polished PowerPoint presentation for
Deep Learning Based Super-Resolution Mapping from Medium Resolution Satellite Imagery
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import re

# Colors (space/satellite theme)
DARK_BLUE = RGBColor(0x0B, 0x2A, 0x4A)
ACCENT_BLUE = RGBColor(0x1E, 0x88, 0xE5)
LIGHT_BLUE = RGBColor(0xE3, 0xF2, 0xFD)
DARK_TEXT = RGBColor(0x1A, 0x1A, 0x2E)
LIGHT_TEXT = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_TEXT = RGBColor(0x55, 0x55, 0x66)
ORANGE = RGBColor(0xFF, 0x8F, 0x00)


def add_background(slide, color):
    """Add a solid color background rectangle."""
    left = top = 0
    width = Inches(13.333)
    height = Inches(7.5)
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # Move to back
    spTree = shape._element.getparent()
    spTree.remove(shape._element)
    spTree.insert(2, shape._element)
    return shape


def add_accent_bar(slide):
    """Add left accent bar."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.25), Inches(7.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_BLUE
    shape.line.fill.background()
    return shape


def add_title(slide, title, top=Inches(0.4), color=DARK_BLUE, size=32):
    """Add a styled title."""
    tb = slide.shapes.add_textbox(Inches(0.6), top, Inches(12.5), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.name = "Calibri"
    run.font.color.rgb = color
    return tb


def add_underline(slide, top=Inches(1.25)):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), top, Inches(1.5), Inches(0.06))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_BLUE
    line.line.fill.background()


def add_body(slide, lines, top=Inches(1.5), left=Inches(0.6), width=Inches(12.1), height=Inches(5.5), size=16):
    """Add body content with bullet-style formatting."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)

        # Detect headings vs bullets
        is_math = line.startswith("$$") or line.endswith("$$")
        is_heading = (len(line) < 60 and line.endswith((":", "?")) 
                      and not line.startswith("•"))
        is_arrow_flow = "→" in line or "↓" in line or "▼" in line or "│" in line
        is_code_flow = any(ch in line for ch in ["┌", "└", "┐", "┘", "─", "│"])

        run = p.add_run()
        # Clean math markers
        clean = line.replace("$$", "").replace("\\rightarrow", "→").replace("\\times", "×")
        clean = clean.replace("\\to", "→").replace("\\approx", "≈")
        run.text = clean
        run.font.name = "Consolas" if (is_code_flow or is_arrow_flow) else "Calibri"
        run.font.size = Pt(12 if is_code_flow else size)
        run.font.color.rgb = DARK_TEXT
        if is_heading:
            run.font.bold = True
            run.font.color.rgb = ACCENT_BLUE
            run.font.size = Pt(size + 2)
        if is_math:
            run.font.italic = True
            run.font.color.rgb = ORANGE
            run.font.bold = True
    return tb


def add_footer(slide, page_num, total):
    """Add page number footer."""
    tb = slide.shapes.add_textbox(Inches(12.3), Inches(7.05), Inches(1), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = f"{page_num} / {total}"
    run.font.size = Pt(10)
    run.font.color.rgb = GRAY_TEXT
    run.font.name = "Calibri"

    tb2 = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(6), Inches(0.3))
    tf2 = tb2.text_frame
    p2 = tf2.paragraphs[0]
    run2 = p2.add_run()
    run2.text = "LDSR-S2 · Sentinel-2 Super-Resolution"
    run2.font.size = Pt(10)
    run2.font.color.rgb = GRAY_TEXT
    run2.font.name = "Calibri"


def create_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)

    # Decorative accent bars
    bar1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.4), Inches(1.5), Inches(0.08))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = ACCENT_BLUE
    bar1.line.fill.background()

    # Main title
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Deep Learning Based Super-Resolution Mapping"
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = LIGHT_TEXT
    run.font.name = "Calibri"

    p2 = tf.add_paragraph()
    run2 = p2.add_run()
    run2.text = "from Medium Resolution Satellite Imagery"
    run2.font.size = Pt(32)
    run2.font.bold = True
    run2.font.color.rgb = ACCENT_BLUE
    run2.font.name = "Calibri"

    # Subtitle
    sub = slide.shapes.add_textbox(Inches(0.8), Inches(4.7), Inches(11.7), Inches(0.6))
    sub_tf = sub.text_frame
    sp = sub_tf.paragraphs[0]
    sr = sp.add_run()
    sr.text = "Using Latent Diffusion for Sentinel-2 Super-Resolution"
    sr.font.size = Pt(20)
    sr.font.italic = True
    sr.font.color.rgb = LIGHT_BLUE
    sr.font.name = "Calibri"

    # Meta info box
    meta = slide.shapes.add_textbox(Inches(0.8), Inches(5.5), Inches(11.7), Inches(1.3))
    meta_tf = meta.text_frame
    meta_tf.word_wrap = True
    info = [
        "Model: LDSR-S2   |   Input: Sentinel-2 RGB + NIR",
        "Input: 10 m  →  Output: 2.5 m   |   Super-resolution factor: 4×",
        "Framework: PyTorch / OpenSR   |   Deployment: Kaggle GPU + Tesla T4",
    ]
    for i, ln in enumerate(info):
        pi = meta_tf.paragraphs[0] if i == 0 else meta_tf.add_paragraph()
        ri = pi.add_run()
        ri.text = ln
        ri.font.size = Pt(14)
        ri.font.color.rgb = LIGHT_BLUE
        ri.font.name = "Calibri"

    # Presenter
    pres = slide.shapes.add_textbox(Inches(0.8), Inches(6.85), Inches(11.7), Inches(0.4))
    pres_tf = pres.text_frame
    pp = pres_tf.paragraphs[0]
    pr = pp.add_run()
    pr.text = "Presented by: Shreeyansh Janu"
    pr.font.size = Pt(14)
    pr.font.bold = True
    pr.font.color.rgb = LIGHT_TEXT
    pr.font.name = "Calibri"


def parse_content(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    slides = []
    current = None
    for line in lines:
        line = line.rstrip("\n")
        stripped = line.strip()
        m = re.match(r"Slide\s+(\d+)\s*—\s*(.+)", stripped)
        if m:
            if current:
                slides.append(current)
            current = {
                "number": int(m.group(1)),
                "title": m.group(2).strip(),
                "content": [],
            }
        elif current and stripped:
            current["content"].append(stripped)
    if current:
        slides.append(current)
    return slides


def create_content_slide(prs, slide_data, page, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, RGBColor(0xFA, 0xFB, 0xFD))
    add_accent_bar(slide)

    add_title(slide, slide_data["title"], top=Inches(0.4))
    add_underline(slide, top=Inches(1.15))

    # Filter/limit content lines to fit slide
    content = slide_data["content"]
    # Heuristic: max ~18 lines per slide, truncate long lines
    trimmed = []
    for c in content[:22]:
        if len(c) > 120:
            c = c[:117] + "..."
        trimmed.append(c)

    # Adjust font size based on amount of content
    size = 16 if len(trimmed) <= 12 else (14 if len(trimmed) <= 18 else 12)
    add_body(slide, trimmed, top=Inches(1.4), size=size)
    add_footer(slide, page, total)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    content_path = r"C:\Users\arthu\.qoder\tmp\C--Users-arthu-Desktop-satellite-srm\attachments\7d1a0a2a-b55b-484a-b87a-ee9d3ba5a75a\78f833db-b9a6-4095-bdf9-854053bd0247.txt"
    slides_content = parse_content(content_path)

    # Title slide first (from slide 1 data)
    create_title_slide(prs)

    # Remaining slides (skip slide 1 since we made a custom title)
    remaining = [s for s in slides_content if s["number"] != 1]
    total_pages = len(remaining) + 1

    for i, sd in enumerate(remaining, start=2):
        create_content_slide(prs, sd, i, total_pages)

    output = r"C:\Users\arthu\Desktop\satellite-srm\LDSR-S2_Presentation.pptx"
    prs.save(output)
    print(f"Presentation saved to: {output}")
    print(f"Total slides: {total_pages}")


if __name__ == "__main__":
    main()
