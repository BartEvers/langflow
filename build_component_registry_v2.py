
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# -------------------------- Type normalization -------------------------------
TYPE_NORM = {
    "string": "str",
    "text": "str",
    "prompt": "str",       # UI prompt fields accept text
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "boolean": "bool",
    "message": "Message",
    "chatmessage": "Message",
    "basemessage": "Message",
    "humanmessage": "Message",
    "aimessage": "Message",
    "toolmessage": "Message",
    "document": "Document",
    "list[document]": "Document[]",
    "list[message]": "Message[]",
    "list[str]": "str[]",
    "any": "any",
}

def norm_type(t: Any) -> str:
    if t is None:
        return ""
    if isinstance(t, dict):
        # sometimes types are objects like {"type":"str"}
        t = t.get("type") or t.get("_type") or ""
    s = str(t).strip()
    key = s.lower()
    return TYPE_NORM.get(key, s)

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

# -------------------------- Extraction helpers -------------------------------

def extract_inputs_from_top_level(data: dict) -> Dict[str, List[str]]:
    out = {}
    inp = data.get("inputs")
    if isinstance(inp, list):
        for item in inp:
            name = item.get("name")
            if not name: continue
            types = item.get("input_types") or item.get("types")
            out[name] = [norm_type(t) for t in ensure_list(types)]
    elif isinstance(inp, dict):
        for name, cfg in inp.items():
            if isinstance(cfg, dict):
                types = cfg.get("input_types") or cfg.get("types")
                out[name] = [norm_type(t) for t in ensure_list(types)]
            else:
                out[name] = [norm_type(t) for t in ensure_list(cfg)]
    return out

def extract_outputs_from_top_level(data: dict) -> Dict[str, List[str]]:
    out = {}
    outs = data.get("outputs")
    if isinstance(outs, list):
        for item in outs:
            name = item.get("name")
            if not name: continue
            types = item.get("types")
            if not types and item.get("selected"):
                types = [item.get("selected")]
            out[name] = [norm_type(t) for t in ensure_list(types)]
    elif isinstance(outs, dict):
        for name, cfg in outs.items():
            if isinstance(cfg, dict):
                types = cfg.get("types")
                if not types and cfg.get("selected"):
                    types = [cfg.get("selected")]
                out[name] = [norm_type(t) for t in ensure_list(types)]
            else:
                out[name] = [norm_type(t) for t in ensure_list(cfg)]
    return out

def extract_from_node_templates(data: dict) -> Dict[str, List[str]]:
    out = {}
    node = data.get("node") or {}
    templ = node.get("template") or {}
    if isinstance(templ, dict):
        for name, cfg in templ.items():
            if not isinstance(cfg, dict): 
                continue
            types = cfg.get("input_types") or cfg.get("types")
            # Some fields (like template/input on Prompt) may have "type":"prompt" but no input_types: treat as str
            if not types and cfg.get("type") in ("prompt", "string", "str", "text"):
                types = ["str"]
            out[name] = [norm_type(t) for t in ensure_list(types)]
    return out

def extract_from_node_outputs(data: dict) -> Dict[str, List[str]]:
    out = {}
    node = data.get("node") or {}
    outs = node.get("outputs") or []
    if isinstance(outs, list):
        for item in outs:
            name = item.get("name")
            if not name: 
                continue
            types = item.get("types")
            if not types and item.get("selected"):
                types = [item.get("selected")]
            out[name] = [norm_type(t) for t in ensure_list(types)]
    return out

def merge_io(pref: Dict[str, List[str]], alt: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = dict(pref)  # start with preferred (node-level usually)
    for k, v in alt.items():
        if k not in out or not out[k]:
            out[k] = v
    return out

def normalize_component_type(obj: dict) -> str:
    t = obj.get("type") or ""
    if t:
        return str(t)
    dn = obj.get("display_name") or obj.get("name") or "UnknownComponent"
    return str(dn).replace(" ", "")

def extract_prompt_variables(data: dict) -> list[str]:
    import re
    node = data.get("node") or {}
    templ = node.get("template") or {}
    vars_found = set()
    for k, cfg in templ.items():
        if isinstance(cfg, dict) and isinstance(cfg.get("value"), str):
            for m in re.findall(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", cfg["value"]):
                vars_found.add(m)
    return sorted(vars_found)

# -------------------------- Overrides ----------------------------------------

DEFAULT_OVERRIDES = {
    "Prompt": {
        "inputs": {"template": ["str"], "input": ["str"], "tool_placeholder": ["Message"]},
        "outputs": {"prompt": ["Message"]},
    },
    "PromptTemplate": {
        "inputs": {"template": ["str"], "input": ["str"], "tool_placeholder": ["Message"]},
        "outputs": {"prompt": ["Message"]},
    },
    "TextInput": {
        "inputs": {"input_value": ["str"]},
        "outputs": {"text": ["str"]},
    },
    "LanguageModelComponent": {
        "inputs": {"system_message": ["Message"], "user_message": ["Message"], "input_value": ["Message"]},
        "outputs": {"text_output": ["Message"]},
    },
}

def deep_merge_types(base: Dict[str, List[str]], override: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = dict(base)
    for k, v in override.items():
        if not out.get(k):
            out[k] = v
    return out

def apply_overrides(entry: dict, overrides: dict) -> dict:
    t = entry["type"]
    # try both type and display_name (normalized without spaces)
    disp = (entry.get("display_name") or "").replace(" ", "")
    o = overrides.get(t) or overrides.get(disp) or {}
    if not o:
        return entry
    entry["inputs"]  = deep_merge_types(entry.get("inputs", {}),  o.get("inputs", {}))
    entry["outputs"] = deep_merge_types(entry.get("outputs", {}), o.get("outputs", {}))
    return entry

# -------------------------- Build Registry -----------------------------------

def build_entry_from_json(path: Path, overrides: dict) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Skipping {path.name}: invalid JSON ({e})", file=sys.stderr)
        return None

    ctype = normalize_component_type(data)
    display = data.get("display_name") or ctype

    node_inp  = extract_from_node_templates(data)
    node_out  = extract_from_node_outputs(data)
    top_inp   = extract_inputs_from_top_level(data)
    top_out   = extract_outputs_from_top_level(data)

    inputs  = merge_io(node_inp, top_inp)
    outputs = merge_io(node_out, top_out)

    entry = {
        "type": ctype,
        "display_name": display,
        "inputs": inputs,
        "outputs": outputs,
    }

    vars_ = extract_prompt_variables(data)
    if vars_:
        entry["variables"] = vars_

    entry = apply_overrides(entry, overrides)

    if data.get("documentation"):
        entry["documentation"] = data["documentation"]
    if data.get("source"):
        entry["source"] = data["source"]

    return entry

def build_registry(dir_path: Path, overrides_path: Path | None, pretty: bool) -> dict:
    overrides = DEFAULT_OVERRIDES.copy()
    if overrides_path and overrides_path.exists():
        try:
            ext = json.loads(overrides_path.read_text(encoding="utf-8"))
            for k, v in ext.items():
                cur = overrides.get(k, {})
                merged = {
                    "inputs":  {**cur.get("inputs", {}),  **v.get("inputs", {})},
                    "outputs": {**cur.get("outputs", {}), **v.get("outputs", {})},
                }
                overrides[k] = merged
        except Exception as e:
            print(f"[WARN] Failed to read overrides {overrides_path}: {e}", file=sys.stderr)

    components = []
    for p in sorted(dir_path.glob("*.json")):
        entry = build_entry_from_json(p, overrides)
        if entry:
            components.append(entry)

    return {
        "registry_version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(dir_path),
        "components": components
    }

def main():
    ap = argparse.ArgumentParser(description="Build a Langflow component handle registry with proper types.")
    ap.add_argument("--dir", required=True, help="Directory containing component JSON files.")
    ap.add_argument("--out", default="component_registry.json", help="Output registry path (JSON).")
    ap.add_argument("--overrides", default="", help="Optional JSON file with type overrides per component type.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = ap.parse_args()

    dir_path = Path(args.dir).expanduser()
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(2)

    overrides_path = Path(args.overrides).expanduser() if args.overrides else None
    registry = build_registry(dir_path, overrides_path, args.pretty)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, separators=(",", ":"), ensure_ascii=False)

    print(f"[OK] Wrote registry: {out_path} (components: {len(registry['components'])})")

if __name__ == "__main__":
    main()
