#!/usr/bin/env python3
"""
Test script for the new prompt composition endpoint.
This will show you exactly what gets sent to the LLM without actually calling it.
"""

import requests
import json
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:7860"  # Adjust if your Langflow runs on a different port
AIRELIUS_BASE = f"{BASE_URL}/api/v1/airelius"

def test_prompt_composition(prompt: str, flow_id: str = None, k: int = 8) -> Dict[str, Any]:
    """Test prompt composition without calling the LLM."""
    print(f"🔍 Testing prompt composition for: '{prompt[:50]}...'")
    
    try:
        payload = {
            "prompt": prompt,
            "k": k
        }
        
        if flow_id:
            payload["flow_id"] = flow_id
            print(f"   🔄 Flow ID: {flow_id}")
        
        response = requests.post(f"{AIRELIUS_BASE}/index/test-prompt-composition", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            if "error" not in data:
                print(f"✅ Prompt composition successful!")
                print(f"   📊 Total length: {data['prompt_analysis']['total_length']} characters")
                print(f"   📄 Total lines: {data['prompt_analysis']['total_lines']}")
                print(f"   🔄 Flow data included: {data['flow_info']['flow_data_included']}")
                print(f"   📋 Snippets retrieved: {data['retrieval_info']['retrieved_count']}")
                
                # Show the complete composed prompt
                print(f"\n{'='*80}")
                print(f"🚀 COMPLETE COMPOSED PROMPT")
                print(f"{'='*80}")
                print(data['composed_prompt'])
                print(f"{'='*80}")
                
                # Show prompt analysis
                print(f"\n📊 PROMPT ANALYSIS:")
                sections = data['prompt_analysis']['sections']
                for section, present in sections.items():
                    status = "✅" if present else "❌"
                    print(f"   {status} {section}")
                
                # Show content breakdown
                print(f"\n📏 CONTENT BREAKDOWN:")
                breakdown = data['prompt_analysis']['content_breakdown']
                for key, value in breakdown.items():
                    if value >= 0:
                        print(f"   {key}: {value} characters")
                    else:
                        print(f"   {key}: Not found")
                
                # Show flow info if available
                if data['flow_info']['flow_data_included']:
                    flow_summary = data['flow_info']['flow_summary']
                    print(f"\n🔄 FLOW SUMMARY:")
                    print(f"   Nodes: {flow_summary['nodes_count']}")
                    print(f"   Edges: {flow_summary['edges_count']}")
                    print(f"   Node types: {', '.join(flow_summary['node_types'])}")
                
                # Show snippets preview
                print(f"\n📋 SNIPPETS PREVIEW:")
                for i, snippet in enumerate(data['retrieval_info']['snippets_preview']):
                    print(f"   Snippet {i+1}:")
                    print(f"     ID: {snippet['id']}")
                    print(f"     Type: {snippet['meta_type']}")
                    print(f"     Preview: {snippet['text_preview']}")
                
                return data
            else:
                print(f"❌ Error in prompt composition: {data['error']}")
                return data
        else:
            print(f"❌ Failed to compose prompt: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"❌ Error testing prompt composition: {e}")
        return {}

def main():
    """Main test function."""
    print("🚀 Airelius Prompt Composition Testing")
    print("=" * 60)
    print("This endpoint shows exactly what gets sent to the LLM")
    print("without actually calling it - perfect for debugging!")
    print("=" * 60)
    
    # Test 1: Basic prompt without flow
    print("\n1️⃣ Testing basic prompt composition...")
    test_prompt_composition("How to create a chat component with memory")
    
    # Test 2: Prompt with specific component focus
    print("\n2️⃣ Testing component-specific prompt...")
    test_prompt_composition("Create a RAG pipeline with document loading and vector search")
    
    # Test 3: Prompt with flow context (if you have a flow ID)
    print("\n3️⃣ Testing prompt with flow context...")
    print("   Note: Set a valid flow_id below to test with flow context")
    # test_prompt_composition("Add error handling to my existing flow", flow_id="your-flow-id-here")
    
    print("\n✅ Testing completed!")
    print("\n📚 Available endpoints:")
    print(f"   POST {AIRELIUS_BASE}/index/test-prompt-composition")
    print(f"   POST {AIRELIUS_BASE}/index/test-snippets")
    print(f"   POST {AIRELIUS_BASE}/index/compare-prompts")
    print(f"   GET  {AIRELIUS_BASE}/index/status")

if __name__ == "__main__":
    main()
