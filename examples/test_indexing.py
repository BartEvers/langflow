#!/usr/bin/env python3
"""
Test script to verify component indexing works.
"""

import sys
from pathlib import Path

# Add the backend to the path
backend_path = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_path))
base_path = backend_path / "base"
sys.path.insert(0, str(base_path))

def test_indexing():
    """Test the indexing functionality."""
    try:
        from langflow.airelius.service import PFUService
        from langflow.airelius.retriever import Retriever
        
        print("✅ Successfully imported modules")
        
        # Create service and retriever
        service = PFUService()
        retriever = service.retriever
        
        print(f"✅ Created service and retriever")
        print(f"   Current count: {retriever.count()}")
        
        # Test with some dummy data
        test_docs = [
            {
                "id": "test:component1",
                "text": "This is a test component for testing purposes",
                "meta": {"type": "test", "name": "component1"}
            },
            {
                "id": "test:component2", 
                "text": "Another test component to verify indexing works",
                "meta": {"type": "test", "name": "component2"}
            }
        ]
        
        print(f"   Testing with {len(test_docs)} test documents...")
        
        # Try to upsert
        result = retriever.upsert(test_docs)
        print(f"   Upsert result: {result}")
        
        # Check count again
        new_count = retriever.count()
        print(f"   New count: {new_count}")
        
        if new_count > 0:
            print("✅ Indexing successful!")
            
            # Try a query
            results = retriever.query("test component", k=2)
            print(f"   Query results: {len(results)} items")
            for i, result in enumerate(results):
                print(f"      Result {i+1}: {result.get('text', '')[:50]}...")
        else:
            print("❌ Indexing failed - count still 0")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Component Indexing")
    print("=" * 40)
    test_indexing()
