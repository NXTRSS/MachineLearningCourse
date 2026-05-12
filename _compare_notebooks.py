#!/usr/bin/env python3
"""Compare two Jupyter notebooks cell by cell and report ALL differences."""

import json
import difflib
import sys


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_text(cell):
    """Return the source of a cell as a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def main():
    path1 = "/Users/kamiljedryczek/Documents/ALK/MachineLearningCodes/MachineLearningCourse/Function_Calling_final4_parser.ipynb"
    path2 = "/Users/kamiljedryczek/Documents/ALK/MachineLearningCodes/MachineLearningCourse/Function_Calling_final4_chat_pass.ipynb"

    nb1 = load_notebook(path1)
    nb2 = load_notebook(path2)

    cells1 = nb1["cells"]
    cells2 = nb2["cells"]

    print(f"Notebook A (parser):    {len(cells1)} cells")
    print(f"Notebook B (chat_pass): {len(cells2)} cells")
    print("=" * 80)

    # --- Compare metadata ---
    meta_diffs = []
    for key in set(list(nb1.get("metadata", {}).keys()) + list(nb2.get("metadata", {}).keys())):
        v1 = nb1.get("metadata", {}).get(key)
        v2 = nb2.get("metadata", {}).get(key)
        if v1 != v2:
            meta_diffs.append((key, v1, v2))
    if meta_diffs:
        print("\n*** NOTEBOOK-LEVEL METADATA DIFFERENCES ***")
        for key, v1, v2 in meta_diffs:
            print(f"  Key: {key}")
            print(f"    A: {v1}")
            print(f"    B: {v2}")
    else:
        print("\nNotebook-level metadata: IDENTICAL")

    # --- Compare cells ---
    max_cells = max(len(cells1), len(cells2))
    diff_count = 0
    source_diff_count = 0

    for i in range(max_cells):
        if i >= len(cells1):
            diff_count += 1
            print(f"\n{'='*80}")
            print(f"CELL {i}: EXISTS ONLY IN B (chat_pass)")
            print(f"  Type: {cells2[i].get('cell_type', '?')}")
            src = source_text(cells2[i])
            preview = src[:200] + ("..." if len(src) > 200 else "")
            print(f"  Preview: {preview}")
            continue
        if i >= len(cells2):
            diff_count += 1
            print(f"\n{'='*80}")
            print(f"CELL {i}: EXISTS ONLY IN A (parser)")
            print(f"  Type: {cells1[i].get('cell_type', '?')}")
            src = source_text(cells1[i])
            preview = src[:200] + ("..." if len(src) > 200 else "")
            print(f"  Preview: {preview}")
            continue

        c1 = cells1[i]
        c2 = cells2[i]

        # Check cell type
        type1 = c1.get("cell_type", "")
        type2 = c2.get("cell_type", "")

        src1 = source_text(c1)
        src2 = source_text(c2)

        # Check cell-level metadata (cell_type, id, metadata dict)
        id1 = c1.get("id", "")
        id2 = c2.get("id", "")

        # Check outputs
        out1 = c1.get("outputs", [])
        out2 = c2.get("outputs", [])

        # Check execution_count
        ec1 = c1.get("execution_count")
        ec2 = c2.get("execution_count")

        # Source diff
        source_differs = (src1 != src2)
        type_differs = (type1 != type2)
        id_differs = (id1 != id2)
        output_differs = (json.dumps(out1) != json.dumps(out2))
        ec_differs = (ec1 != ec2)

        if source_differs or type_differs:
            diff_count += 1
            if source_differs:
                source_diff_count += 1

            print(f"\n{'='*80}")
            print(f"CELL {i}: DIFFERS")
            if type_differs:
                print(f"  Cell type: A={type1}, B={type2}")
            else:
                print(f"  Cell type: {type1}")

            if source_differs:
                lines1 = src1.splitlines(keepends=True)
                lines2 = src2.splitlines(keepends=True)
                diff = list(difflib.unified_diff(
                    lines1, lines2,
                    fromfile=f"A (parser) cell {i}",
                    tofile=f"B (chat_pass) cell {i}",
                    lineterm=""
                ))
                if diff:
                    print("  SOURCE DIFF:")
                    for line in diff:
                        print(f"    {line.rstrip()}")

            if id_differs:
                print(f"  ID: A={id1}, B={id2}")

    # Also report cells with only output/execution_count differences (less important but noted)
    output_only_diffs = 0
    for i in range(min(len(cells1), len(cells2))):
        c1 = cells1[i]
        c2 = cells2[i]
        src1 = source_text(c1)
        src2 = source_text(c2)
        if src1 == src2 and c1.get("cell_type") == c2.get("cell_type"):
            out1 = c1.get("outputs", [])
            out2 = c2.get("outputs", [])
            ec1 = c1.get("execution_count")
            ec2 = c2.get("execution_count")
            id1 = c1.get("id", "")
            id2 = c2.get("id", "")
            if json.dumps(out1) != json.dumps(out2) or ec1 != ec2 or id1 != id2:
                output_only_diffs += 1

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"  Total cells compared: {max_cells}")
    print(f"  Cells with SOURCE or TYPE differences: {diff_count} ({source_diff_count} source diffs)")
    print(f"  Cells with ONLY output/exec_count/id differences (same source): {output_only_diffs}")
    print(f"  Completely identical cells (source + type): {max_cells - diff_count}")


if __name__ == "__main__":
    main()
