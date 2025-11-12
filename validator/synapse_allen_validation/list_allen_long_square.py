# list_allen_long_square.py
from allensdk.core.cell_types_cache import CellTypesCache

ctc = CellTypesCache(manifest_file="allen_cache/manifest.json")

cells = ctc.get_cells()
cands = []
for c in cells:
    try:
        # Heuristics: mouse, excitatory/pyramidal-ish, has long-square sweeps
        if (c.get("species") == "Mus musculus" and
            ( (c.get("dendrite_type") or "").lower() in ("spiny", "pyramidal") or
              (c.get("class_label") or "").lower() in ("excitatory","pyramidal") )):
            sweeps = ctc.get_ephys_sweeps(c["id"])
            if any("long" in (s.get("stimulus_name","").lower()) for s in sweeps):
                cands.append({
                    "specimen_id": c["id"],
                    "dendrite_type": c.get("dendrite_type"),
                    "cre_line": c.get("cre_line"),
                    "structure": c.get("structure__acronym"),
                })
    except Exception:
        pass

# show a few
cands = sorted(cands, key=lambda x: x["specimen_id"])[:20]
for x in cands:
    print(x)
print(f"\nFound {len(cands)} candidates shown (more exist; adjust slicing as needed).")
