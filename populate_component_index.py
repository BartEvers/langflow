#!/usr/bin/env python3
"""
Script to populate the Langflow component index with actual component data.
This script scans the components directory and extracts real component information.
"""

import json
import os
import ast
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional

def extract_component_info(file_path: Path) -> Optional[Dict[str, Any]]:
    """Extract component information from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        component_info = {
            "file_path": str(file_path),
            "classes": [],
            "imports": [],
            "docstring": None
        }
        
        # Extract module docstring
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
            component_info["docstring"] = tree.body[0].value.s
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    component_info["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    component_info["imports"].append(f"{module}.{alias.name}")
        
        # Extract class information
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "attributes": []
                }
                
                # Extract methods and attributes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info["methods"].append({
                            "name": item.name,
                            "docstring": ast.get_docstring(item)
                        })
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)
                
                component_info["classes"].append(class_info)
        
        return component_info
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def scan_components_directory(components_dir: Path) -> Dict[str, Any]:
    """Scan the components directory and extract component information."""
    components_data = {}
    
    if not components_dir.exists():
        print(f"Components directory not found: {components_dir}")
        return components_data
    
    for category_dir in components_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('_'):
            category_name = category_dir.name
            components_data[category_name] = {
                "description": f"Components in {category_name} category",
                "base_path": str(category_dir),
                "components": {}
            }
            
            # Scan Python files in this category
            for py_file in category_dir.glob("*.py"):
                if not py_file.name.startswith('_'):
                    component_name = py_file.stem
                    component_info = extract_component_info(py_file)
                    if component_info:
                        components_data[category_name]["components"][component_name] = component_info
    
    return components_data

def scan_template_directory(template_dir: Path) -> Dict[str, Any]:
    """Scan the template directory and extract template information."""
    template_data = {}
    
    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}")
        return template_data
    
    # Scan key template files
    key_files = [
        "template/base.py",
        "frontend_node/base.py", 
        "field/base.py"
    ]
    
    for file_path in key_files:
        full_path = template_dir / file_path
        if full_path.exists():
            component_info = extract_component_info(full_path)
            if component_info:
                template_data[file_path.replace('/', '_').replace('.py', '')] = component_info
    
    return template_data

def scan_base_directory(base_dir: Path) -> Dict[str, Any]:
    """Scan the base directory and extract base class information."""
    base_data = {}
    
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return base_data
    
    # Look for base component files
    for py_file in base_dir.rglob("*.py"):
        if not py_file.name.startswith('_'):
            component_info = extract_component_info(py_file)
            if component_info:
                base_data[str(py_file)] = component_info
    
    return base_data

def main():
    """Main function to populate the component index."""
    print("Starting component index population...")
    
    # Load the base index
    with open('langflow_component_index.json', 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Define directories to scan
    base_path = Path("src/backend/base/langflow")
    components_dir = base_path / "components"
    template_dir = base_path / "template"
    base_dir = base_path / "base"
    custom_dir = base_path / "custom"
    
    print(f"Scanning components directory: {components_dir}")
    components_data = scan_components_directory(components_dir)
    
    print(f"Scanning template directory: {template_dir}")
    template_data = scan_template_directory(template_dir)
    
    print(f"Scanning base directory: {base_dir}")
    base_data = scan_base_directory(base_dir)
    
    # Update the index with real data
    index["component_categories"].update(components_data)
    
    # Update template structure with real data
    for key, data in template_data.items():
        if key in index["template_structure"]:
            index["template_structure"][key].update(data)
    
    # Update base classes with real data
    for key, data in base_data.items():
        if "base_classes" in key:
            # Extract class name from file path
            class_name = key.split('/')[-1].replace('.py', '')
            if class_name in index["base_classes"]:
                index["base_classes"][class_name].update(data)
    
    # Count total components
    total_components = sum(
        len(cat.get("components", {})) 
        for cat in index["component_categories"].values()
    )
    index["metadata"]["total_components"] = total_components
    
    # Save the updated index
    output_file = "langflow_component_index_populated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"Component index populated successfully!")
    print(f"Total components found: {total_components}")
    print(f"Output saved to: {output_file}")
    
    # Print summary
    print("\nComponent categories found:")
    for category, data in components_data.items():
        component_count = len(data.get("components", {}))
        print(f"  {category}: {component_count} components")

if __name__ == "__main__":
    main()
