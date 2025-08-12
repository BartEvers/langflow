#!/usr/bin/env python3
"""
Quick test script for the new snippets retrieval endpoints.
Run this to quickly verify the endpoints are working.
"""

import requests
import sys

def quick_test():
    """Quick test of the new endpoints."""
    base_url = "http://localhost:7860"
    airelius_base = f"{base_url}/api/v1/airelius"
    
    print("🚀 Quick Test of Airelius Snippets Retrieval Endpoints")
    print("=" * 60)
    
    # Test 1: Index Status
    print("\n1️⃣ Testing Index Status...")
    try:
        response = requests.get(f"{airelius_base}/index/status")
        if response.status_code == 200:
            data = response.json()
            doc_count = data.get('index_status', {}).get('total_documents', 0)
            mode = data.get('index_status', {}).get('mode', 'unknown')
            print(f"   ✅ Status: {doc_count} documents, Mode: {mode}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Test Snippets Retrieval
    print("\n2️⃣ Testing Snippets Retrieval...")
    try:
        payload = {
            "prompt": "test prompt for retrieval",
            "k": 3,
            "include_samples": True,
            "include_debug": False
        }
        response = requests.post(f"{airelius_base}/index/test-snippets", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                retrieved = data.get('retrieved_count', 0)
                print(f"   ✅ Retrieved {retrieved} snippets")
            else:
                print(f"   ⚠️  Warning: {data['error']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Compare Prompts
    print("\n3️⃣ Testing Prompt Comparison...")
    try:
        payload = {
            "prompts": ["test", "component", "flow"],
            "k": 2
        }
        response = requests.post(f"{airelius_base}/index/compare-prompts", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                total = data.get('total_snippets_retrieved', 0)
                print(f"   ✅ Compared prompts, total snippets: {total}")
            else:
                print(f"   ⚠️  Warning: {data['error']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Quick test completed!")
    print("\n📚 Available endpoints:")
    print(f"   GET  {airelius_base}/index/status")
    print(f"   POST {airelius_base}/index/test-snippets")
    print(f"   POST {airelius_base}/index/compare-prompts")
    print(f"   POST {airelius_base}/index/test-query")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
