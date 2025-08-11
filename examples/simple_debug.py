#!/usr/bin/env python3
"""
Simple debug script for the Airelius PFU system.
This script examines the files directly without importing modules.
"""

import os
import json
import sqlite3
from pathlib import Path

def check_vector_db():
    """Check the vector database directly."""
    print("🔍 Checking Vector Database")
    print("=" * 40)
    
    db_path = Path("data/airelius_index/chroma.sqlite3")
    if not db_path.exists():
        print(f"❌ Vector database not found at: {db_path}")
        return
    
    print(f"✅ Vector database found at: {db_path}")
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Database tables: {[table[0] for table in tables]}")
        
        # Check embeddings table
        if ('embeddings',) in tables:
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"📈 Total embeddings: {count}")
            
            if count > 0:
                # Get a sample
                cursor.execute("SELECT id, embedding_id, document, metadata FROM embeddings LIMIT 3;")
                samples = cursor.fetchall()
                print(f"📝 Sample embeddings:")
                for i, sample in enumerate(samples):
                    doc_id, emb_id, doc, metadata = sample
                    print(f"   {i+1}. ID: {doc_id}, Doc: {doc[:100]}...")
                    if metadata:
                        try:
                            meta = json.loads(metadata)
                            print(f"      Metadata: {meta}")
                        except:
                            print(f"      Metadata: {metadata}")
        
        # Check documents table if it exists
        if ('documents',) in tables:
            cursor.execute("SELECT COUNT(*) FROM documents;")
            count = cursor.fetchone()[0]
            print(f"📄 Total documents: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

def check_component_files():
    """Check if component files exist and are accessible."""
    print("\n🔍 Checking Component Files")
    print("=" * 40)
    
    # Check if the airelius directory exists
    airelius_path = Path("src/backend/base/langflow/airelius")
    if airelius_path.exists():
        print(f"✅ Airelius directory found: {airelius_path}")
        
        # List files in the directory
        files = list(airelius_path.glob("*.py"))
        print(f"📁 Python files: {[f.name for f in files]}")
        
        # Check specific files
        for file_name in ["service.py", "router.py", "retriever.py", "kernel.py"]:
            file_path = airelius_path / file_name
            if file_path.exists():
                print(f"✅ {file_name} exists")
                
                # Check file size
                size = file_path.stat().st_size
                print(f"   Size: {size} bytes")
                
                # Check if file is readable
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                        print(f"   First line: {first_line}")
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")
            else:
                print(f"❌ {file_name} not found")
    else:
        print(f"❌ Airelius directory not found: {airelius_path}")

def check_example_files():
    """Check the example files."""
    print("\n🔍 Checking Example Files")
    print("=" * 40)
    
    examples_path = Path("examples")
    if examples_path.exists():
        print(f"✅ Examples directory found: {examples_path}")
        
        # List example files
        example_files = list(examples_path.glob("*.py"))
        print(f"📁 Example files: {[f.name for f in example_files]}")
        
        # Check specific files
        for file_name in ["pfu_usage_example.py", "test_pfu_system.py"]:
            file_path = examples_path / file_name
            if file_path.exists():
                print(f"✅ {file_name} exists")
                
                # Check file size
                size = file_path.stat().st_size
                print(f"   Size: {size} bytes")
            else:
                print(f"❌ {file_name} not found")
    else:
        print(f"❌ Examples directory not found: {examples_path}")

def check_environment():
    """Check the environment setup."""
    print("🌍 Environment Check")
    print("=" * 30)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the langflow project root
    if (current_dir / "src" / "backend" / "base" / "langflow").exists():
        print("✅ Langflow project structure detected")
    else:
        print("❌ Not in langflow project root")
    
    # Check for virtual environment
    venv_path = current_dir / ".venv"
    if venv_path.exists():
        print(f"✅ Virtual environment found: {venv_path}")
        
        # Check if it's activated
        if "VIRTUAL_ENV" in os.environ:
            print(f"✅ Virtual environment is active: {os.environ['VIRTUAL_ENV']}")
        else:
            print("⚠️ Virtual environment exists but not active")
    else:
        print("❌ No virtual environment found")
    
    # Check for dependency files
    dep_files = ["pyproject.toml", "uv.lock", "requirements.txt"]
    for dep_file in dep_files:
        if (current_dir / dep_file).exists():
            print(f"✅ {dep_file} found")
        else:
            print(f"❌ {dep_file} not found")

def main():
    """Main debug function."""
    print("🚀 Starting Simple PFU System Debug")
    print("=" * 50)
    
    check_environment()
    check_component_files()
    check_vector_db()
    check_example_files()
    
    print("\n💡 Debug Summary:")
    print("1. Check if virtual environment is activated")
    print("2. Verify vector database has indexed components")
    print("3. Ensure all required files exist and are readable")
    print("4. Check for any syntax errors in Python files")

if __name__ == "__main__":
    main()
