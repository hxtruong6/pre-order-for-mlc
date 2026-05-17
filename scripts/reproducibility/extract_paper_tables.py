#!/usr/bin/env python3
"""Extract hardcoded pgfplots numbers from result tables in paper.tex.

Usage: python extract_paper_tables.py <start_line> <end_line> <out_csv>
Lines are 1-indexed and inclusive, matching the \\begin{table}..\\end{table} span.

Output columns: dataset, metric, direction, classifier, alpha, value
"""
import re
import sys
from pathlib import Path

CLASSIFIERS = {
    1: "PA-H-2", 2: "PA-H", 3: "PA-S-2", 4: "PA-S",
    5: "PR-H-2", 6: "PR-H", 7: "PR-S-2", 8: "PR-S",
    9: "BR", 10: "CC", 11: "CLR",
}
ALPHAS = [0.0, 0.1, 0.2, 0.3]
COLOR_TO_ALPHA = {"red": 0.0, "blue": 0.1, "green!60!black": 0.2, "cyan": 0.3}

METRIC_PATTERNS = [
    (r"f_\{\\text\{MLC\}\}\^\{1\}", "F1"),
    (r"f_\{\\text\{MLC\}\}\^\{\\text\{ham\}\}", "Hamming"),
    (r"f_\{\\text\{MLC\}\}\^\{\\text\{sub\}\}", "Subset"),
    (r"\bAFRD\b", "AFRD"),
    (r"\bMFRD\b", "MFRD"),
    (r"\bAABS\b", "AABS"),
    (r"\bABS\b", "ABS"),
]

def parse_label_line(line):
    """Split a tabular row label line into per-column (dataset, metric, direction)."""
    line = line.strip().rstrip("\\").strip()
    cells = [c.strip() for c in line.split("&")]
    out = []
    for cell in cells:
        if not cell:
            out.append((None, None, None))
            continue
        # Dataset name is the prefix before ':'
        m_ds = re.match(r"([A-Za-z0-9_\-]+)\s*:", cell)
        dataset = m_ds.group(1) if m_ds else None
        metric = None
        for pat, name in METRIC_PATTERNS:
            if re.search(pat, cell):
                metric = name
                break
        direction = None
        if "uparrow" in cell:
            direction = "up"
        elif "downarrow" in cell:
            direction = "down"
        out.append((dataset, metric, direction))
    return out


def parse_tikz_cell(block):
    """Parse one tikzpicture block. Returns list of (alpha, classifier_idx, value)."""
    results = []
    # Find each addplot color block
    # Pattern: \addplot[<color>, ...] coordinates { (x,y) (x,y) ... };
    for m in re.finditer(
        r"\\addplot\[([^\]]+)\]\s*coordinates\s*\{([^}]*)\}",
        block,
        re.DOTALL,
    ):
        opts = m.group(1)
        coords_str = m.group(2)
        # Identify color
        color = None
        for c in COLOR_TO_ALPHA:
            if re.search(r"\b" + re.escape(c) + r"\b", opts):
                color = c
                break
        if color is None:
            continue  # grid (white) or other
        alpha = COLOR_TO_ALPHA[color]
        for cm in re.finditer(r"\(\s*(\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", coords_str):
            idx = int(cm.group(1))
            val = float(cm.group(2))
            results.append((alpha, idx, val))
    return results


def split_into_cells(table_body):
    """Split table body into tikzpicture blocks paired with the row label line that follows.

    Returns a list of (cell_index_in_row, dataset, metric, direction, tikz_block).
    """
    # Find all tikzpicture spans
    tikz_spans = []
    for m in re.finditer(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", table_body, re.DOTALL):
        tikz_spans.append((m.start(), m.end(), m.group(1)))

    # Find all row label lines: lines that contain `\eqref` and end with `\\`
    # We treat each `\\` that occurs OUTSIDE a tikzpicture as a row separator.
    # Easier: split table_body on `\\\\` and find segments that are between two tikz groups.
    # Use a different approach: walk through the body, accumulating tikz cells (in row order),
    # then when we hit `\\` outside any tikz, emit a row with the next label line.

    # Strategy: identify positions of `\\` (row-terminator) outside tikzpicture; the immediately
    # preceding tabular row consists of N tikz cells, and the label line follows.
    # In this paper layout, the label line *follows* `\\` ? Looking at the example:
    #   ... \end{tikzpicture} } \\
    #         GpositivePse: ... & plantPse: ... & HumanPse: ... \\
    # So: row-of-tikz, then `\\`, then label line, then `\\`.

    # Mask out tikz regions so we can find row terminators.
    masked = list(table_body)
    for s, e, _ in tikz_spans:
        for i in range(s, e):
            masked[i] = " "
    masked = "".join(masked)

    # Find positions of `\\` row terminators in masked text.
    row_term_positions = [m.start() for m in re.finditer(r"\\\\", masked)]

    # Walk: between consecutive row terminators is either a row of tikz cells or a label line.
    cells_out = []
    prev = 0
    pending_tikz_row = []  # list of tikz_block strings
    for pos in row_term_positions:
        segment = table_body[prev:pos]
        masked_seg = masked[prev:pos]
        # Determine: does this segment contain tikz cells, or is it a label line?
        # Count tikz spans intersecting [prev, pos)
        contained = [(s, e, b) for (s, e, b) in tikz_spans if s >= prev and e <= pos]
        if contained:
            # This is a tikz row; remember in order
            pending_tikz_row = [b for (_, _, b) in sorted(contained)]
        else:
            # This is (likely) a label line
            stripped = segment.strip()
            if stripped and "&" in stripped and pending_tikz_row:
                labels = parse_label_line(stripped + "\\\\")
                for i, (block, lab) in enumerate(zip(pending_tikz_row, labels)):
                    ds, metric, direction = lab
                    cells_out.append((i, ds, metric, direction, block))
                pending_tikz_row = []
        prev = pos + 2  # skip past `\\`
    return cells_out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    start_line = int(sys.argv[1])
    end_line = int(sys.argv[2])
    out_csv = Path(sys.argv[3])

    paper = Path("/Users/xuantruong/Documents/WORK/RESEARCH/preorders4MLC/paper.tex")
    lines = paper.read_text().splitlines()
    body = "\n".join(lines[start_line - 1 : end_line])

    cells = split_into_cells(body)
    print(f"Found {len(cells)} cells", file=sys.stderr)

    rows = []
    for col_idx, ds, metric, direction, block in cells:
        if ds is None or metric is None:
            print(f"WARN: missing dataset/metric for cell col={col_idx}", file=sys.stderr)
        triples = parse_tikz_cell(block)
        for alpha, idx, val in triples:
            classifier = CLASSIFIERS.get(idx)
            if classifier is None:
                continue
            rows.append((ds, metric, direction, classifier, alpha, val))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("dataset,metric,direction,classifier,alpha,value\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Wrote {len(rows)} rows to {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
