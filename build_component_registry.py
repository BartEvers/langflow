#!/usr/bin/env python3
"""
Build a component handle registry from Langflow component JSON files.

Usage:
  python build_component_registry.py \
    --dir "/Applications/XAMPP/xamppfiles/htdocs/aimable/langflow/index_payload" \
    --out "./component_registry.json"

What it does:
- Reads every *.json in --dir
- Extracts, per component type:
    - inputs: { <fieldName>: [<type> ...] }  (from `input_types` if present)
    - outputs: { <name>: [<type> ...] }      (from `types` or `selected` if present)
- Writes a compact registry JSON with version + timestamp
- Never "invents" handle names; if types are missing, leaves them empty []
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# ---- Helpers -----------------------------------------------------------------

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def extract_input_types(inp: dict) -> list:
    """
    Prefer explicit `input_types` if present; otherwise return [] (unknown).
    Do NOT guess types; the registry is a source of truth for handle names.
    """
    types = inp.get("input_types")
    if types is None:
        return []
    return [str(t) for t in ensure_list(types)]

def extract_output_types(out: dict) -> list:
    """
    Prefer `types` if present, else `selected`. If neither present, [] (unknown).
    """
    if "types" in out and isinstance(out["types"], list):
        return [str(t) for t in out["types"]]
    if "selected" in out and out["selected"]:
        return [str(out["selected"])]
    return []

def normalize_component_type(obj: dict, fallback: str = None) -> str:
    # Prefer "type" (e.g., "TextInput", "LanguageModelComponent")
    ctype = obj.get("type") or fallback
    if not ctype:
        # as a last resort try display_name with no spaces
        ctype = (obj.get("display_name") or "UnknownComponent").replace(" ", "")
    return str(ctype)

def build_entry_from_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Skipping {path.name}: invalid JSON ({e})", file=sys.stderr)
        return None

    ctype = normalize_component_type(data)
    display = data.get("display_name") or ctype
    # Inputs
    inputs = {}
    for inp in ensure_list(data.get("inputs")):
        name = inp.get("name")
        if not name:
            continue
        inputs[name] = extract_input_types(inp)

    # Outputs
    outputs = {}
    for out in ensure_list(data.get("outputs")):
        name = out.get("name")
        if not name:
            continue
        outputs[name] = extract_output_types(out)

    # Minimal record
    return {
        "type": ctype,
        "display_name": display,
        "inputs": inputs,     # { fieldName: [types] }
        "outputs": outputs,   # { handleName: [types] }
        # Optional references that might help debugging:
        "documentation": data.get("documentation") or "",
        "source": data.get("source") or "",
    }

def build_registry(dir_path: Path) -> dict:
    components = []
    for p in sorted(dir_path.glob("*.json")):
        entry = build_entry_from_json(p)
        if entry:
            components.append(entry)

    registry = {
        "registry_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(dir_path),
        "components": components
    }
    return registry

# ---- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a Langflow component handle registry.")
    ap.add_argument("--dir", required=True, help="Directory containing component JSON files.")
    ap.add_argument("--out", default="component_registry.json", help="Output registry path (JSON).")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = ap.parse_args()

    dir_path = Path(args.dir).expanduser()
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(2)

    registry = build_registry(dir_path)
    out_path = Path(args.out)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            else:
                json.dump(registry, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"[OK] Wrote registry: {out_path}  (components: {len(registry['components'])})")

if __name__ == "__main__":
    main()
