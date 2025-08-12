#!/usr/bin/env python3
"""
Extract structured metadata from a Langflow-like repo into JSON "cards"
suitable for indexing in a vector DB.

- Components & Bundles  -> Component Cards
- Inputs                -> Input Field Dictionaries
- IO (ports/outputs)    -> Port/Output Dictionaries
- Base                  -> Framework Rules (light summary)
- Utils/constants.py    -> Constants/Enums Dictionaries
- Schema                -> Schema Cards

No imports/side effects: uses Python AST only.

Usage:
  python extract_langflow_metadata.py

Adjust ROOTS if your repo layout differs.
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# CONFIG
# ----------------------------
ROOTS = [
    "src/backend/base/langflow/components",
    "src/backend/base/langflow/bundles",
    "src/backend/base/langflow/io",
    "src/backend/base/langflow/inputs",
    "src/backend/base/langflow/base",
    "src/backend/base/langflow/utils",
    "src/backend/base/langflow/typing",
    "src/backend/base/langflow/schema",
]

OUT_DIR = Path("index_payload")
SUBDIRS = {
    "component": OUT_DIR / "components",
    "input_field": OUT_DIR / "input_fields",
    "port_spec": OUT_DIR / "ports",
    "framework_rules": OUT_DIR / "framework",
    "enum_constants": OUT_DIR / "constants",
    "schema": OUT_DIR / "schema",
}

# ----------------------------
# AST UTILS
# ----------------------------

def get_basename(node: ast.AST) -> str:
    """Return base name for ast.Name/ast.Attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{get_basename(node.value)}.{node.attr}"
    return type(node).__name__


def ast_literal(node: ast.AST) -> Any:
    """Best-effort conversion of AST node into a Python literal (safe)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):  # <= py3.7
        return node.s
    if isinstance(node, ast.Num):  # <= py3.7
        return node.n
    if isinstance(node, ast.NameConstant):  # <= py3.8
        return node.value
    if isinstance(node, ast.List):
        return [ast_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(ast_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {ast_literal(k): ast_literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # handle negative numbers: -1
        v = ast_literal(node.operand)
        try:
            return -v
        except Exception:
            return None
    if isinstance(node, ast.Name):
        # Return the symbol name (useful for enums/constants)
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{ast_literal(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        # Represent simple constructor calls as strings, e.g., "Path('...')"
        func = get_basename(node.func)
        args = ", ".join(repr(ast_literal(a)) for a in node.args)
        kwargs = ", ".join(f"{k.arg}={repr(ast_literal(k.value))}" for k in node.keywords if k.arg)
        inside = ", ".join(x for x in [args, kwargs] if x)
        return f"{func}({inside})"
    # Fallback to string repr of node type
    return f"<{type(node).__name__}>"


def class_bases(cls: ast.ClassDef) -> List[str]:
    return [get_basename(b) for b in cls.bases]


def is_component_class(cls: ast.ClassDef) -> bool:
    # Heuristic: any class with a base ending in 'Component'
    return any(base.split(".")[-1].endswith("Component") for base in class_bases(cls))


def is_input_class(cls: ast.ClassDef) -> bool:
    # Heuristic: class name endswith 'Input' OR bases contain *Input
    if cls.name.endswith("Input"):
        return True
    return any(base.split(".")[-1].endswith("Input") for base in class_bases(cls))


def is_output_like_class(cls: ast.ClassDef) -> bool:
    # Heuristic: class name == 'Output' OR base endswith 'Output'
    if cls.name == "Output":
        return True
    return any(base.split(".")[-1].endswith("Output") for base in class_bases(cls))


def find_assign_list(cls: ast.ClassDef, varname: str) -> Optional[List[ast.AST]]:
    """Find a class-level assignment like `inputs = [ ... ]` and return elements."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == varname:
                    if isinstance(stmt.value, (ast.List, ast.Tuple)):
                        return list(stmt.value.elts)
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == varname:
            val = stmt.value
            if isinstance(val, (ast.List, ast.Tuple)):
                return list(val.elts)
    return None


def get_class_attr(cls: ast.ClassDef, attr_name: str) -> Tuple[Optional[Any], Optional[int]]:
    """Return (value, lineno) for a simple class attribute (constant)."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == attr_name:
                    return ast_literal(stmt.value), stmt.lineno
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
            return ast_literal(stmt.value), stmt.lineno
    return None, None


def parse_call_kwargs(node: ast.AST) -> Tuple[str, Dict[str, Any]]:
    """For a Call node, return (callable_name, kwargs_dict)."""
    if not isinstance(node, ast.Call):
        return (get_basename(node), {})
    fn = get_basename(node.func)
    kwargs = {}
    # positional args recorded as _arg0, _arg1 if needed
    for i, a in enumerate(node.args):
        kwargs[f"_arg{i}"] = ast_literal(a)
    for kw in node.keywords:
        if kw.arg:
            kwargs[kw.arg] = ast_literal(kw.value)
    return fn, kwargs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(obj: Dict[str, Any], path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# EXTRACTORS
# ----------------------------

def extract_components_from_file(py_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef) and is_component_class(node):
            cls = node
            # core attributes
            comp_name = get_class_attr(cls, "name")[0] or cls.name
            display_name = get_class_attr(cls, "display_name")[0] or cls.name
            description = get_class_attr(cls, "description")[0] or (ast.get_docstring(cls) or "")
            documentation = get_class_attr(cls, "documentation")[0]
            icon = get_class_attr(cls, "icon")[0]
            minimized = get_class_attr(cls, "minimized")[0]
            tags = get_class_attr(cls, "tags")[0]

            # inputs & outputs
            inputs_list = []
            raw_inputs = find_assign_list(cls, "inputs") or []
            for elt in raw_inputs:
                call_name, kwargs = parse_call_kwargs(elt)
                # normalize some common fields
                input_entry = {
                    "field_class": call_name,
                    **kwargs
                }
                # normalize synonyms for defaults
                if "value" in input_entry and "default" not in input_entry:
                    input_entry["default"] = input_entry["value"]
                inputs_list.append(input_entry)

            outputs_list = []
            raw_outputs = find_assign_list(cls, "outputs") or []
            for elt in raw_outputs:
                call_name, kwargs = parse_call_kwargs(elt)
                outputs_list.append({
                    "output_class": call_name,
                    **kwargs
                })

            template_keys = [i.get("name") for i in inputs_list if isinstance(i.get("name"), str)]

            comp = {
                "kind": "component",
                "type": comp_name,
                "display_name": display_name,
                "description": description,
                "documentation": documentation,
                "icon": icon,
                "minimized": minimized,
                "tags": tags if isinstance(tags, (list, tuple)) else (tags or []),
                "inputs": inputs_list,
                "outputs": outputs_list,
                "template": template_keys,
                "examples": [
                    {
                        "op": "add_component",
                        "type": comp_name
                    }
                ],
                "source": f"{py_path.as_posix()}#L{cls.lineno}"
            }
            out.append(comp)
    return out


def extract_input_fields_from_file(py_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef) and is_input_class(node):
            cls = node
            # find __init__ to capture supported kwargs
            kwargs: List[str] = []
            for stmt in cls.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    args = stmt.args
                    # capture keyword-only and positional-or-keyword (excluding self, *args/**kwargs)
                    names = [a.arg for a in args.args if a.arg not in ("self",)]
                    kwonly = [a.arg for a in args.kwonlyargs]
                    # ignore vararg/kwarg catch-alls
                    kwargs = sorted(set(names + kwonly))
                    break
            entry = {
                "kind": "input_field",
                "input_class": cls.name,
                "supported_kwargs": kwargs,
                "source": f"{py_path.as_posix()}#L{cls.lineno}"
            }
            out.append(entry)
    return out


def extract_ports_from_file(py_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef) and is_output_like_class(node):
            cls = node
            fields: List[str] = []
            # inspect __init__ signature if present
            for stmt in cls.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    args = stmt.args
                    names = [a.arg for a in args.args if a.arg not in ("self",)]
                    kwonly = [a.arg for a in args.kwonlyargs]
                    fields = sorted(set(names + kwonly))
                    break
            entry = {
                "kind": "port_spec",
                "class": cls.name,
                "fields": fields,
                "source": f"{py_path.as_posix()}#L{cls.lineno}"
            }
            out.append(entry)
    return out


def extract_framework_rules_from_file(py_path: Path, tree: ast.AST) -> Optional[Dict[str, Any]]:
    """Create a lightweight summary of base classes and common attributes."""
    base_classes = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef):
            bases = class_bases(node)
            for b in bases:
                if b.endswith("Component") and node.name not in base_classes:
                    base_classes.append(node.name)
    if not base_classes:
        return None
    return {
        "kind": "framework_rules",
        "file": py_path.as_posix(),
        "base_component_classes_in_file": base_classes,
        "expected_attributes_hint": ["name", "display_name", "description", "documentation", "icon", "inputs", "outputs"],
    }


def extract_constants_from_file(py_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
    """Uppercase simple assignments -> constant groups."""
    out = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            # only simple NAME targets; ignore tuple unpack, etc.
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key.isupper():
                    val = ast_literal(node.value)
                    # We only keep simple serializable things
                    if isinstance(val, (str, int, float, list, tuple, dict, bool, type(None))):
                        out.append({
                            "kind": "enum_constants",
                            "name": key,
                            "value": val,
                            "source": f"{py_path.as_posix()}#L{node.lineno}"
                        })
    return out


def extract_schema_from_file(py_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
    """Summarize classes as schema-ish cards: name + annotated fields + docstring."""
    out = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef):
            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            entry = {
                "kind": "schema",
                "name": node.name,
                "fields": sorted(set(fields)),
                "doc": ast.get_docstring(node) or "",
                "source": f"{py_path.as_posix()}#L{node.lineno}"
            }
            out.append(entry)
    return out


# ----------------------------
# DRIVER
# ----------------------------

def process_file(py_path: Path):
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    except Exception as e:
        print(f"[parse-error] {py_path}: {e}")
        return

    rel = py_path.as_posix()

    # components & bundles
    if "/components/" in rel or "/bundles/" in rel:
        comps = extract_components_from_file(py_path, tree)
        for c in comps:
            fname = re.sub(r"[^\w\-]+", "_", c["type"]) or "component"
            write_json(c, SUBDIRS["component"] / f"{fname}.json")

    # inputs
    if "/inputs/" in rel:
        inputs = extract_input_fields_from_file(py_path, tree)
        for item in inputs:
            fname = item["input_class"]
            write_json(item, SUBDIRS["input_field"] / f"{fname}.json")

    # io (ports)
    if "/io/" in rel:
        ports = extract_ports_from_file(py_path, tree)
        for item in ports:
            fname = item["class"]
            write_json(item, SUBDIRS["port_spec"] / f"{fname}.json")

    # base (framework rules)
    if "/base/" in rel and "/base/langflow/base/" in rel:
        rules = extract_framework_rules_from_file(py_path, tree)
        if rules:
            fname = Path(rel).stem
            write_json(rules, SUBDIRS["framework_rules"] / f"{fname}.json")

    # constants
    if rel.endswith("/utils/constants.py") or rel.endswith("\\utils\\constants.py"):
        consts = extract_constants_from_file(py_path, tree)
        if consts:
            # Write each constant separately for granular retrieval
            for c in consts:
                fname = c["name"]
                write_json(c, SUBDIRS["enum_constants"] / f"{fname}.json")

    # schema
    if "/schema/" in rel:
        schemas = extract_schema_from_file(py_path, tree)
        for s in schemas:
            fname = s["name"]
            write_json(s, SUBDIRS["schema"] / f"{fname}.json")


def walk_and_process(roots: List[str]):
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for py_path in path.rglob("*.py"):
            process_file(py_path)


def main():
    for p in SUBDIRS.values():
        ensure_dir(p)
    walk_and_process(ROOTS)
    print(f"Done. Wrote JSON cards to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()