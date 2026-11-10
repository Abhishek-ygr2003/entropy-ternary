# -*- coding: utf-8 -*-
"""Verify round-3 resume PDFs: content + page count + text order (headings before bullets)."""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")
from pypdf import PdfReader

FILES = {
    "AI/ML": r"C:\Users\2026\Downloads\Abhishek_Yadav_GR_Resume_AI_ML_DS.pdf",
    "FoSSE": r"C:\Users\2026\Downloads\Abhishek_Yadav_GR_Resume_Microsoft_FoSSE.pdf",
}
# heading -> text that must appear AFTER it (first bullet of that section)
ORDER = {
    "AI/ML": [
        ("CodeT5+ Code-Quality", "10,000 code samples"),
        ("AI Business Intelligence Copilot", "entirely offline"),
        ("py2rs", "zero external crate dependencies"),
        ("CareerLens", "retry logic"),
    ],
    "FoSSE": [
        ("py2rs / rustify", "zero external crate dependencies"),
        ("OrchestrAI", "prompt-based generation of machine-generated bug predictions"),
        ("CareerLens", "retry logic"),
    ],
}
PRESENT_AI = ["1 point above the best corrected", "Retracted", "not resolved between 0.2 and 0.3", "zeroing mask"]
PRESENT_FOSSE = ["1 point above the best corrected", "Retracted", "not resolved between 0.2 and 0.3",
                 "machine-generated bug prediction", "512-token context"]
ABSENT_BOTH = ["2 points above", "Killed what", "scaling artifact", "severity classification",
               "static/dynamic bug detection", "automated bug detection"]

fail = 0
for name, path in FILES.items():
    r = PdfReader(path)
    pages = len(r.pages)
    t = "\n".join(p.extract_text() for p in r.pages)
    n = t.replace("-\n", "").replace("\n", " ")
    print(f"=== {name}: {pages} pages ===")
    if pages != 2:
        print(f"FAIL page count {pages}"); fail += 1
    for term in (PRESENT_AI if name == "AI/ML" else PRESENT_FOSSE):
        ok = term in n
        if not ok: fail += 1
        print(("OK  " if ok else "FAIL"), "present:", repr(term))
    for term in ABSENT_BOTH:
        ok = term not in n
        if not ok: fail += 1
        print(("OK  " if ok else "FAIL"), "absent :", repr(term))
    for head, after in ORDER[name]:
        i = n.find(head)
        j = n.find(after)
        ok = i != -1 and j != -1 and i < j
        if not ok: fail += 1
        print(("OK  " if ok else "FAIL"), f"order: {head!r} (idx {i}) before {after!r} (idx {j})")
print("---", "ALL OK" if fail == 0 else f"{fail} FAILURES")
