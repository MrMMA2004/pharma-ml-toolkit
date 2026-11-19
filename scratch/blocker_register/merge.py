#!/usr/bin/env python3
"""Merge cover + body into the final single PDF (Report brief pipeline)."""
from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89


def normalize(page):
    box = page.mediabox
    w, h = float(box.width), float(box.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        page.scale_to(A4_W, A4_H)
    return page


writer = PdfWriter()
cover = PdfReader("cover.pdf").pages[0]
writer.add_page(normalize(cover))
for p in PdfReader("body.pdf").pages:
    writer.add_page(normalize(p))
writer.add_metadata({
    "/Title": "PharmaML Studio part11 Blocker Register",
    "/Author": "Z.ai",
    "/Creator": "Z.ai",
    "/Subject": ("Which parts of MASTER_PROMPT part11_1 and part11_2 "
                 "cannot be executed in this environment, and why"),
})
out = "/home/z/my-project/PharmaML_Studio_part11_Blocker_Register.pdf"
with open(out, "wb") as f:
    writer.write(f)
print("final:", out, "pages:", len(writer.pages))
