# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
p = r"C:\Users\2026\Downloads\entropy_ternary_paper.tex"
t = open(p, encoding="utf-8").read()
fixes = [
    (r"\bibitem[Li and Liu(2016)]{li2016ternary}", r"\bibitem[Li et~al.(2016)]{li2016ternary}"),
    ("F.~Li and B.~Liu,", "F.~Li, B.~Zhang, and B.~Liu,"),
]
for old, new in fixes:
    n = t.count(old)
    print(f"{n}x  {old[:60]}")
    assert n == 1, f"unexpected count {n} for: {old[:60]}"
    t = t.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(t)
print("public tex TWN fixes done")
