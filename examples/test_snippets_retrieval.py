#!/usr/bin/env python3
"""
Test script for the new snippets retrieval endpoints in the Airelius system.

This script demonstrates how to test the vector database retrieval functionality
using the new endpoints we created.
"""

import requests
import json
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:7860"  # Adjust if your Langflow runs on a different port
AIRELIUS_BASE = f"{BASE_URL}/api/v1/airelius"

def test_index_status() -> Dict[str, Any]:
    """Test the index status endpoint."""
    print("🔍 Testing index status...")
    
    try:
        response = requests.get(f"{AIRELIUS_BASE}/index/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Index status: {data['index_status']['total_documents']} documents")
            print(f"   Mode: {data['index_status']['mode']}")
            print(f"   Ready: {data['index_status']['index_ready']}")
            return data
        else:
            print(f"❌ Failed to get index status: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error testing index status: {e}")
        return {}

def test_snippets_retrieval(prompt: str, k: int = 5) -> Dict[str, Any]:
    """Test snippets retrieval with a specific prompt."""
    print(f"\n🔍 Testing snippets retrieval for: '{prompt[:50]}...'")
    
    try:
        payload = {
            "prompt": prompt,
            "k": k,
            "include_samples": True,
            "include_debug": True
        }
        
        response = requests.post(f"{AIRELIUS_BASE}/index/test-snippets", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                print(f"✅ Retrieved {data['retrieved_count']} snippets")
                print(f"   Total documents: {data['current_count']}")
                print(f"   Retrieval percentage: {data['retrieval_summary']['retrieved_percentage']}%")
                
                # Show snippet previews
                for i, snippet in enumerate(data['retrieved_snippets'][:3]):
                    print(f"   Snippet {i+1}: {snippet.get('text', '')[:100]}...")
                
                return data
            else:
                print(f"❌ Error in retrieval: {data['error']}")
                return data
        else:
            print(f"❌ Failed to retrieve snippets: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error testing snippets retrieval: {e}")
        return {}

def test_prompt_comparison(prompts: List[str], k: int = 3) -> Dict[str, Any]:
    """Test retrieval performance across multiple prompts."""
    print(f"\n🔍 Testing prompt comparison for {len(prompts)} prompts...")
    
    try:
        payload = {
            "prompts": prompts,
            "k": k
        }
        
        response = requests.post(f"{AIRELIUS_BASE}/index/compare-prompts", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                print(f"✅ Comparison completed successfully")
                print(f"   Total snippets retrieved: {data['total_snippets_retrieved']}")
                print(f"   Average per prompt: {data['retrieval_quality_metrics']['average_retrieved_per_prompt']}")
                
                # Show results by prompt
                for prompt, result in data['results_by_prompt'].items():
                    print(f"   '{prompt[:30]}...': {result['retrieved_count']} snippets")
                
                # Show overlap analysis
                if data['overlap_analysis']:
                    print("   Overlap analysis:")
                    for comparison, overlap in data['overlap_analysis'].items():
                        print(f"     {comparison}: {overlap['overlap_count']} overlapping ({overlap['overlap_percentage']}%)")
                
                return data
            else:
                print(f"❌ Error in comparison: {data['error']}")
                return data
        else:
            print(f"❌ Failed to compare prompts: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error testing prompt comparison: {e}")
        return {}

def test_specific_use_cases():
    """Test specific use cases that are common in Langflow."""
    print("\n🎯 Testing specific use cases...")
    
    # Test cases for different types of prompts
    test_cases = [
        {
            "name": "Component Search",
            "prompt": "How to create a chat component with memory",
            "k": 5
        },
        {
            "name": "Flow Creation",
            "prompt": "Building a RAG pipeline with document loading",
            "k": 5
        },
        {
            "name": "Tool Integration",
            "prompt": "Integrating external APIs and tools in flows",
            "k": 5
        },
        {
            "name": "Error Handling",
            "prompt": "Handling errors and exceptions in Langflow",
            "k": 5
        }
    ]
    
    results = {}
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        result = test_snippets_retrieval(test_case['prompt'], test_case['k'])
        results[test_case['name']] = result
    
    return results

def main():
    """Main test function."""
    print("🚀 Airelius Snippets Retrieval Testing")
    print("=" * 50)
    
    # Test 1: Check index status
    status = test_index_status()
    if not status or status.get('index_status', {}).get('total_documents', 0) == 0:
        print("\n⚠️  No documents indexed. Please run /index/reload first.")
        return
    
    # Test 2: Basic snippets retrieval
    test_snippets_retrieval("vector database search", 5)
    
    # Test 3: Prompt comparison
    prompts = [
        "component creation",
        "flow building", 
        "API integration",
        "error handling"
    ]
    test_prompt_comparison(prompts, 3)
    
    # Test 4: Specific use cases
    test_specific_use_cases()
    
    print("\n✅ Testing completed!")
    print("\n📚 Available endpoints:")
    print(f"   GET  {AIRELIUS_BASE}/index/status")
    print(f"   POST {AIRELIUS_BASE}/index/test-snippets")
    print(f"   POST {AIRELIUS_BASE}/index/compare-prompts")
    print(f"   POST {AIRELIUS_BASE}/index/test-query")

if __name__ == "__main__":
    main()
