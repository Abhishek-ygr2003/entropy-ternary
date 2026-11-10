# -*- coding: utf-8 -*-
"""Rebuild the anonymized TMLR supplementary zip from the repo working tree.
- adds round-2 control files (script, JSONs, log, extra checkpoints)
- excludes __pycache__/data
- scrubs identity strings from README before packing
- asserts no identity strings remain anywhere in the archive"""
import sys, os, re, zipfile, io
sys.stdout.reconfigure(encoding="utf-8")

REPO = r"C:\Users\2026\entropy-ternary"
OUT = r"C:\Users\2026\Downloads\tmlr_submit\supplementary_anon.zip"

INCLUDE_FILES = [
    "README.md", "requirements.txt",
    "pipeline.py", "experiments_control.py", "experiments_control2.py", "experiments_control3.py",
    "experiments_improvement.py", "plot_results.py", "retrain_checkpoint.py",
    "run_remaining_experiments.py",
    "src/__init__.py", "src/model.py", "src/train.py", "src/ternarize.py",
    "tests/__init__.py", "tests/test_ternarize.py",
]
INCLUDE_DIRS = [
    ("outputs/results", ".json"),
    ("outputs/checkpoints", ".pt"),
]
LOGS = ["outputs/control_run.log", "outputs/control2_run.log", "outputs/control3_run.log"]

IDENTITY = re.compile(r"Abhishek|Yadav|ygr2003|Garden City|abhishek10", re.I)

def scrub(name: str, text: str) -> str:
    if name.endswith("README.md"):
        text = re.sub(r"> \*\*.*?\*\*.*?\n(> .*?\n)*", "", text, count=1)  # title block
        # simpler: remove identity lines individually below
    return text

readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
# Replace the paper-citation block (title/author/university) with an anonymized one
readme = readme.replace(
    "> **Attribution Controls for Neuron Selection in Partial Ternarization: A Negative Result**  \n"
    "> Abhishek Yadav GR  \n"
    "> Garden City University, Bengaluru",
    "> **Attribution Controls for Neuron Selection in Partial Ternarization: A Negative Result**  \n"
    "> (author list withheld for double-blind review)",
)
# Anonymize the bibtex block
readme = readme.replace("yadav2026entropy", "anon2026entropy")
readme = readme.replace("author  = {Abhishek Yadav GR},", "author  = {Anonymous},")
readme = readme.replace("journal = {Preprint (code and results at github.com/Abhishek-ygr2003/entropy-ternary)}",
                        "journal = {Technical report (public repository link withheld for review)}")

entries = []  # (arcname, data)
for f in INCLUDE_FILES:
    if f == "README.md":
        data = readme.encode("utf-8")
    else:
        data = open(os.path.join(REPO, *f.split("/")), "rb").read()
    entries.append((f, data))

for d, ext in INCLUDE_DIRS:
    full = os.path.join(REPO, d.replace("/", os.sep))
    for fn in sorted(os.listdir(full)):
        if fn.endswith(ext):
            entries.append((f"{d}/{fn}", open(os.path.join(full, fn), "rb").read()))

for lg in LOGS:
    data = open(os.path.join(REPO, *lg.split("/")), "rb").read()
    if IDENTITY.search(data.decode("utf-8", errors="replace")):
        raise SystemExit(f"identity string in log: {lg}")
    entries.append((lg, data))

# final anonymity assertion over every text entry
for name, data in entries:
    if name.endswith((".py", ".md", ".txt", ".json", ".log")):
        txt = data.decode("utf-8", errors="replace")
        m = IDENTITY.search(txt)
        if m:
            raise SystemExit(f"IDENTITY LEAK in {name}: {m.group(0)!r}")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name, data in entries:
        z.writestr(name, data)

z = zipfile.ZipFile(OUT)
print(f"{OUT}: {len(z.namelist())} entries, "
      f"{os.path.getsize(OUT)/1e6:.2f} MB")
for n in z.namelist(): print("  ", n)
