#!/usr/bin/env python3
"""Merge cover + body into the final Prompt Registry PDF (strict A4 normalize)."""
from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89
COVER = "/home/z/my-project/scratch/prompt_registry/cover.pdf"
BODY = "/home/z/my-project/scratch/prompt_registry/body.pdf"
OUT = "/home/z/my-project/Prompt_Registry.pdf"


def normalize(page):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        page.scale_to(A4_W, A4_H)
    return page


writer = PdfWriter()
writer.add_page(normalize(PdfReader(COVER).pages[0]))
for p in PdfReader(BODY).pages:
    writer.add_page(normalize(p))
writer.add_metadata({
    "/Title": "The Prompt Registry - Every Instruction Given for the PharmaML Toolkit",
    "/Author": "Z.ai",
    "/Creator": "Z.ai",
    "/Subject": "Complete verified register of every prompt and instruction given by the project owner for the PharmaML Toolkit",
})
with open(OUT, "wb") as f:
    writer.write(f)
print("MERGED:", OUT)
