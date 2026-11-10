# -*- coding: utf-8 -*-
"""Build Downloads public paper (article class, named author) from the
canonical TMLR main.tex body + the old article preamble, with a natbib-
compatible inline bibliography (author-year labels)."""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

OLD = r"C:\Users\2026\Downloads\entropy_ternary_paper.tex"
NEW = r"C:\Users\2026\Downloads\tmlr_submit\tmlr_submission\main.tex"
OUT = r"C:\Users\2026\Downloads\entropy_ternary_paper.tex"

old = open(OLD, encoding="utf-8").read()
main = open(NEW, encoding="utf-8").read()

# ---------- 1. extract old inline bibliography, add author-year labels ----------
labels = {
    "jacob2018quantization":    "Jacob et~al.(2018)",
    "courbariaux2016binarized": "Courbariaux et~al.(2016)",
    "rastegari2016xnor":        "Rastegari et~al.(2016)",
    "li2016ternary":            "Li and Liu(2016)",
    "zhu2017trained":           "Zhu et~al.(2017)",
    "ma2024bitnet":             "Ma et~al.(2024)",
    "nagel2020adaround":        "Nagel et~al.(2020)",
    "han2015deep":              "Han et~al.(2015)",
    "cover2006element":         "Cover and Thomas(2006)",
}
m = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", old, re.S)
assert m, "old bibliography not found"
bib_body = m.group(1)
keys = re.findall(r"\\bibitem\{([^}]+)\}", bib_body)
print("bib keys:", keys)
for k in keys:
    assert k in labels, f"no author-year label for {k}"
    bib_body = bib_body.replace(r"\bibitem{" + k + "}",
                                r"\bibitem[" + labels[k] + "]{" + k + "}")
bib_block = "\\begin{thebibliography}{9}" + bib_body + "\\end{thebibliography}"

# ---------- 2. TMLR body (between \begin{document} and \bibliography) ----------
b0 = main.index(r"\begin{document}")
b1 = main.index(r"\bibliography{main}")
body = main[b0 + len(r"\begin{document}"):b1].rstrip() + "\n\n"

# drop TMLR anonymous author/title remnants if any appear in body (they don't)
assert "Anonymous" not in body

# ---------- 3. assemble preamble ----------
preamble = r"""\documentclass[10pt,letterpaper]{article}
\usepackage{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[round]{natbib}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}

\geometry{margin=1in, top=0.9in, bottom=0.9in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!50!black
}

\title{\bf Attribution Controls for Neuron Selection in Partial Ternarization\\
A Negative Result}

\author{
Abhishek Yadav GR\\
\textit{Garden City University}, Bengaluru, India\\
\texttt{abhishek10ygr@gmail.com}\\
\thanks{Technical report. Code and results: github.com/Abhishek-ygr2003/entropy-ternary.}
}

\date{}

\begin{document}
"""
out = preamble + body.lstrip("\n") + "\n" + bib_block + "\n\n\\end{document}\n"
open(OUT, "w", encoding="utf-8", newline="").write(out)
print("written:", OUT, len(out), "chars")
