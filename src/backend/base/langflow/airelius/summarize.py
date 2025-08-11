from __future__ import annotations

from typing import Any


def build_flow_signature(flow_data: dict | None, *, max_fields_per_node: int = 5) -> dict[str, Any]:
    """Summarize a flow to a compact signature for prompting.

    Keeps id, type, a few template keys, and omits large values.
    """
    if not flow_data:
        return {"nodes": [], "edges": []}
    nodes = []
    for n in flow_data.get("nodes", []):
        node = {
            "id": n.get("id"),
            "type": n.get("type"),
            "display_name": n.get("data", {}).get("display_name"),
        }
        tmpl = n.get("data", {}).get("node", {}).get("template", {})
        # keep a small subset of keys
        small = {}
        for i, (k, v) in enumerate(tmpl.items()):
            if i >= max_fields_per_node:
                break
            if isinstance(v, dict) and "value" in v and not isinstance(v["value"], (dict, list)):
                small[k] = {"value": v["value"]}
            elif k == "_type":
                small[k] = v
        node["template"] = small
        nodes.append(node)

    edges = [
        {"id": e.get("id"), "source": e.get("source"), "target": e.get("target")}
        for e in flow_data.get("edges", [])
    ]
    return {"nodes": nodes, "edges": edges}


