#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example_flows")  # put your 30 flow exports here
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("index_payload/exemplars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sluggify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_.]+", "_", text)
    return re.sub(r"__+", "_", text).strip("_")

def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {p}: {e}")
        return None

def extract_components(flow: dict) -> list[str]:
    """
    Langflow exports vary by version. Try common shapes:
    - nodes: [{ 'data': {'type': 'BingSearchAPIComponent', ...} }]
    - nodes: [{ 'type': 'BingSearchAPIComponent', ...}]
    - components: [{ 'type': '...' }]
    """
    comps = set()
    nodes = flow.get("nodes") or flow.get("components") or []
    for n in nodes:
        t = None
        if isinstance(n, dict):
            if "data" in n and isinstance(n["data"], dict):
                t = n["data"].get("type") or n["data"].get("name")
            t = t or n.get("type") or n.get("name")
        if isinstance(t, str):
            comps.add(t)
    return sorted(comps)

def default_title_from_filename(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return name[:1].upper() + name[1:]

def guess_tags(components: list[str]) -> list[str]:
    tags = set()
    comp_s = " ".join(components).lower()
    if "vector" in comp_s or "retriever" in comp_s: tags.add("rag")
    if "bing" in comp_s or "search" in comp_s: tags.add("websearch")
    if "pdf" in comp_s or "file" in comp_s: tags.add("pdf")
    if "agent" in comp_s or "tool" in comp_s: tags.add("agent")
    if "embed" in comp_s: tags.add("embeddings")
    if "llm" in comp_s or "chat" in comp_s: tags.add("chat")
    return sorted(tags)

def build_exemplar(flow_path: Path) -> dict | None:
    flow = read_json(flow_path)
    if not flow: return None

    components = extract_components(flow)
    title = flow.get("title") or flow.get("name") or default_title_from_filename(flow_path)
    intent = flow.get("intent") or flow.get("description") or []
    if isinstance(intent, str):
        intent = [intent] if intent else []
    tags = flow.get("tags") or guess_tags(components)

    return {
        "kind": "exemplar",
        "title": title,
        "intent": intent,
        "components": components,
        "flow_json": flow,  # keep full export for fidelity
        "notes": flow.get("notes", ""),
        "tags": tags,
        "source": str(flow_path)
    }

def main():
    if not IN_DIR.exists():
        print(f"Input dir not found: {IN_DIR}. Create it and drop your flow JSON files inside.")
        sys.exit(1)

    count = 0
    for p in sorted(IN_DIR.rglob("*.json")):
        ex = build_exemplar(p)
        if not ex: 
            continue
        slug = sluggify(ex["title"] or p.stem) or sluggify(p.stem)
        out = OUT_DIR / f"{slug}.json"
        out.write_text(json.dumps(ex, indent=2, ensure_ascii=False), encoding="utf-8")
        count += 1
        print(f"[ok] {p} -> {out}")
    print(f"Done. Wrote {count} exemplar files to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
