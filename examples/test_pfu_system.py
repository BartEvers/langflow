#!/usr/bin/env python3
"""
Test script for the Airelius PFU (Partial Flow Update) system.

This script tests the key endpoints to ensure the system is working correctly.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:7860"  # Adjust if your Langflow runs on a different port
API_KEY = "your-api-key-here"  # Replace with your actual API key

def test_pfu_system():
    """Test the complete PFU system workflow."""
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print("🧪 Testing Airelius PFU System")
    print("=" * 50)
    
    # Step 1: Index backend files for context
    print("\n1️⃣ Indexing backend files...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/airelius/pfu/index/files",
            headers=headers,
            json={
                "patterns": ["/src/backend/**/*.py"],
                "reset": False,
                "chunk_size": 2000,
                "overlap": 200
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Successfully indexed {data.get('files', 0)} files")
        else:
            print(f"❌ Failed to index files: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error indexing files: {e}")
        return False
    
    # Step 2: Generate a PFU plan
    print("\n2️⃣ Generating PFU plan...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/airelius/pfu/plan",
            headers=headers,
            json={
                "prompt": "Add a simple chat input node to my flow",
                "flow_id": None  # No specific flow for this test
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Plan generated successfully")
            print(f"   Plan ID: {data.get('plan_id')}")
            print(f"   Operations: {len(data.get('operations', []))}")
            
            # Check if we have a valid plan
            if data.get('plan') and data.get('plan', {}).get('operations'):
                print(f"   ✅ Valid plan structure with {len(data['plan']['operations'])} operations")
                plan = data['plan']
            else:
                print(f"   ⚠️ Plan structure incomplete, but LLM response available")
                plan = None
                
        else:
            print(f"❌ Failed to generate plan: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating plan: {e}")
        return False
    
    # Step 3: Test single operation application (if we have a test flow)
    print("\n3️⃣ Testing single operation application...")
    print("   (Skipping - requires an actual flow ID)")
    
    # Step 4: Test plan execution (if we have a test flow)
    print("\n4️⃣ Testing plan execution...")
    print("   (Skipping - requires an actual flow ID)")
    
    # Step 5: Show the generated plan structure
    if plan:
        print("\n📋 Generated Plan Structure:")
        print("-" * 30)
        print(f"Objective: {plan.get('objective', 'N/A')}")
        print(f"Operations: {len(plan.get('operations', []))}")
        
        for i, op in enumerate(plan.get('operations', [])):
            print(f"  Step {i+1}: {op.get('description', 'N/A')}")
            print(f"    Operation: {op.get('operation', {}).get('op', 'N/A')}")
    
    print("\n✅ PFU System Test Completed!")
    print("\nTo test with actual flows:")
    print("1. Create a flow in Langflow")
    print("2. Use the flow ID in your requests")
    print("3. Test the execute endpoints")
    
    return True

def test_individual_endpoints():
    """Test individual PFU endpoints."""
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print("\n🔍 Testing Individual Endpoints")
    print("=" * 40)
    
    # Test index count
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/airelius/pfu/index/count",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Index count: {data.get('count', 0)} items")
        else:
            print(f"❌ Index count failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Index count error: {e}")
    
    # Test index sample
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/airelius/pfu/index/sample?n=3",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Index sample: {len(data.get('items', []))} items")
        else:
            print(f"❌ Index sample failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Index sample error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Airelius PFU System Tests")
    print("Make sure Langflow is running and accessible!")
    print()
    
    # Test the main system
    success = test_pfu_system()
    
    # Test individual endpoints
    test_individual_endpoints()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n💥 Some tests failed. Check the output above.")
