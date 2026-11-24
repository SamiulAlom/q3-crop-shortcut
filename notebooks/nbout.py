"""Print the text outputs of an executed notebook, cell by cell (for reading a run's log folder).
Usage: python notebooks/nbout.py logs/<run>/<notebook>.ipynb [max characters per cell, default 3000] [cells to shorten, e.g. 4,5]
"""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
short = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else set()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    text = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            text.append("ERROR " + o.get("ename", "") + ": " + o.get("evalue", ""))
        elif "text" in o:
            text.append("".join(o["text"]))
    lines = [l for l in "".join(text).splitlines() if not l.startswith("saved ")]
    if i in short:
        lines = lines[-4:]
    if lines:
        print(f"--- cell {i}")
        print("\n".join(lines)[:limit])
