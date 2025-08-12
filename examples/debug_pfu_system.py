#!/usr/bin/env python3
"""
Debug script for the Airelius PFU system.

This script helps diagnose issues with:
1. Vector database retrieval
2. Component indexing
3. LLM response parsing
4. Overall system functionality
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the backend to the path
backend_path = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_path))

# Also add the base langflow path
base_path = backend_path / "base"
sys.path.insert(0, str(base_path))

async def debug_pfu_system():
    """Debug the PFU system step by step."""
    print("🔍 Debugging Airelius PFU System")
    print("=" * 50)
    
    try:
        # Import the necessary modules
        from langflow.airelius.service import PFUService
        from langflow.airelius.retriever import Retriever
        from langflow.interface.components import get_and_cache_all_types_dict
        from langflow.services.deps import get_settings_service
        
        print("✅ Successfully imported required modules")
        
        # Step 1: Check component indexing
        print("\n1️⃣ Checking component indexing...")
        try:
            settings_service = get_settings_service()
            all_types = await get_and_cache_all_types_dict(settings_service=settings_service)
            
            if all_types and "components" in all_types:
                total_components = sum(len(comps) for comps in all_types["components"].values())
                print(f"   ✅ Found {total_components} components in {len(all_types['components'])} categories")
                
                # Show some component categories
                for category, components in list(all_types["components"].items())[:5]:
                    print(f"      {category}: {len(components)} components")
                    if components:
                        # Show first component name
                        first_comp = list(components.keys())[0]
                        print(f"         Example: {first_comp}")
            else:
                print("   ❌ No components found in all_types_dict")
                
        except Exception as e:
            print(f"   ❌ Error checking components: {e}")
        
        # Step 2: Check vector database
        print("\n2️⃣ Checking vector database...")
        try:
            retriever = Retriever()
            count = retriever.count()
            print(f"   ✅ Vector DB has {count} indexed items")
            
            if count > 0:
                # Try a sample query
                sample = retriever.sample(n=3)
                print(f"   ✅ Sample items retrieved: {len(sample)}")
                for i, item in enumerate(sample):
                    print(f"      Item {i+1}: {item.get('text', '')[:80]}...")
            else:
                print("   ⚠️ Vector DB is empty - components may not be indexed")
                
        except Exception as e:
            print(f"   ❌ Error checking vector DB: {e}")
        
        # Step 3: Test PFU service
        print("\n3️⃣ Testing PFU service...")
        try:
            service = PFUService()
            
            # Test component indexing
            if all_types:
                indexed_count = service.index_components(all_types)
                print(f"   ✅ Indexed {indexed_count} components")
                
                # Test retrieval
                test_query = "chat input component"
                retrieved = service.retrieve(test_query, k=3)
                print(f"   ✅ Retrieved {len(retrieved)} items for query: '{test_query}'")
                
                for i, item in enumerate(retrieved):
                    print(f"      Result {i+1}: {item.get('text', '')[:80]}...")
            else:
                print("   ❌ Cannot test service without components")
                
        except Exception as e:
            print(f"   ❌ Error testing PFU service: {e}")
        
        # Step 4: Test prompt composition
        print("\n4️⃣ Testing prompt composition...")
        try:
            if all_types:
                service = PFUService()
                service.index_components(all_types)
                
                # Test prompt composition
                user_prompt = "Add a chat input node to my flow"
                flow_data = {"nodes": [], "edges": []}
                retrieved = service.retrieve(user_prompt, k=3)
                
                composed_prompt = service.compose_prompt(user_prompt, flow_data, retrieved)
                print(f"   ✅ Composed prompt length: {len(composed_prompt)} characters")
                print(f"   ✅ Prompt contains PFU_KERNEL: {'PFU_KERNEL' in composed_prompt}")
                print(f"   ✅ Prompt contains user request: {user_prompt in composed_prompt}")
                print(f"   ✅ Prompt contains component info: {len(retrieved) > 0}")
                
                # Show a snippet of the prompt
                print(f"   📝 Prompt preview (first 200 chars): {composed_prompt[:200]}...")
            else:
                print("   ❌ Cannot test prompt composition without components")
                
        except Exception as e:
            print(f"   ❌ Error testing prompt composition: {e}")
        
        print("\n✅ Debug completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the langflow project root")
        print("   and that all dependencies are installed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def check_environment():
    """Check the environment setup."""
    print("🌍 Environment Check")
    print("=" * 30)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if backend directory exists
    backend_path = current_dir / "src" / "backend"
    if backend_path.exists():
        print(f"✅ Backend directory found: {backend_path}")
    else:
        print(f"❌ Backend directory not found: {backend_path}")
        print("   Make sure you're running this from the langflow project root")
    
    # Check for required files
    required_files = [
        "src/backend/base/langflow/airelius/service.py",
        "src/backend/base/langflow/airelius/router.py",
        "src/backend/base/langflow/airelius/retriever.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

if __name__ == "__main__":
    print("🚀 Starting PFU System Debug")
    print("Make sure you're in the langflow project root directory!")
    print()
    
    # Check environment first
    check_environment()
    print()
    
    # Run the debug
    asyncio.run(debug_pfu_system())
    
    print("\n💡 Debug Tips:")
    print("1. Check the logs for detailed error messages")
    print("2. Ensure all dependencies are installed")
    print("3. Verify the vector database is properly indexed")
    print("4. Check if the LLM API key is set correctly")
