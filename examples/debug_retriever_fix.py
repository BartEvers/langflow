#!/usr/bin/env python3
"""
Debug script to test the fixed retriever.
"""

import sys
from pathlib import Path

# Add the backend to the path
backend_path = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_path))
base_path = backend_path / "base"
sys.path.insert(0, str(base_path))

def test_retriever_fix():
    """Test the retriever with comprehensive logging."""
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
        
        # Test query
        test_query = "test component"
        print(f"   Testing query: '{test_query}'")
        retrieved = service.retrieve(test_query, k=3)
        print(f"   Retrieved {len(retrieved)} results")
        
        for i, item in enumerate(retrieved):
            print(f"      Result {i+1}: {item.get('text', '')[:80]}...")
        
        # Test reload
        print("   Testing document reload...")
        reloaded_count = retriever.reload_documents()
        print(f"   Reloaded {reloaded_count} documents")
        
        # Test query again after reload
        retrieved_after_reload = service.retrieve(test_query, k=3)
        print(f"   After reload, retrieved {len(retrieved_after_reload)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing retriever: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Retriever Fix")
    print("=" * 50)
    
    success = test_retriever_fix()
    
    if success:
        print("\n✅ Retriever test completed successfully!")
    else:
        print("\n❌ Retriever test failed!")
        sys.exit(1)
