#!/usr/bin/env python3
"""
Example script demonstrating how to use the PFU (Partial Flow Update) system.

This shows the complete workflow:
1. Plan generation with AI
2. Step-by-step execution
3. Progress tracking and validation
"""

import asyncio
import json
from typing import Dict, Any

# Example PFU plan structure
EXAMPLE_PLAN = {
    "objective": "Add a simple chat flow with memory",
    "current_state_analysis": "Empty flow with no nodes",
    "required_changes": [
        "Add a chat input node",
        "Add a memory component", 
        "Add a language model node",
        "Add a chat output node",
        "Connect the components in sequence"
    ],
    "execution_strategy": "Build incrementally, adding one component at a time",
    "validation_strategy": "Verify each component is properly positioned and connected",
    "operations": [
        {
            "step": 1,
            "description": "Add chat input node",
            "validation": "Check if chat input node was added with correct position",
            "operation": {
                "op": "add_node",
                "node": {
                    "id": "chat-input-1",
                    "type": "genericNode",
                    "position": {"x": 100, "y": 200},
                    "data": {
                        "id": "chat-input-1",
                        "type": "ChatInput",
                        "display_name": "Chat Input",
                        "node": {
                            "base_classes": ["ChatInput"],
                            "template": {
                                "_type": "ChatInput"
                            }
                        }
                    }
                }
            }
        },
        {
            "step": 2,
            "description": "Add memory component",
            "validation": "Check if memory component was added and positioned correctly",
            "operation": {
                "op": "add_node",
                "node": {
                    "id": "memory-1",
                    "type": "genericNode",
                    "position": {"x": 300, "y": 200},
                    "data": {
                        "id": "memory-1",
                        "type": "ConversationBufferMemory",
                        "display_name": "Memory",
                        "node": {
                            "base_classes": ["ConversationBufferMemory"],
                            "template": {
                                "_type": "ConversationBufferMemory"
                            }
                        }
                    }
                }
            }
        },
        {
            "step": 3,
            "description": "Add language model node",
            "validation": "Check if LLM node was added with proper configuration",
            "operation": {
                "op": "add_node",
                "node": {
                    "id": "llm-1",
                    "type": "genericNode",
                    "position": {"x": 500, "y": 200},
                    "data": {
                        "id": "llm-1",
                        "type": "ChatOpenAI",
                        "display_name": "OpenAI Chat",
                        "node": {
                            "base_classes": ["ChatOpenAI"],
                            "template": {
                                "_type": "ChatOpenAI",
                                "model_name": "gpt-3.5-turbo"
                            }
                        }
                    }
                }
            }
        },
        {
            "step": 4,
            "description": "Add chat output node",
            "validation": "Check if output node was added",
            "operation": {
                "op": "add_node",
                "node": {
                    "id": "chat-output-1",
                    "type": "genericNode",
                    "position": {"x": 700, "y": 200},
                    "data": {
                        "id": "chat-output-1",
                        "type": "ChatOutput",
                        "display_name": "Chat Output",
                        "node": {
                            "base_classes": ["ChatOutput"],
                            "template": {
                                "_type": "ChatOutput"
                            }
                        }
                    }
                }
            }
        },
        {
            "step": 5,
            "description": "Connect chat input to memory",
            "validation": "Check if edge was created between chat input and memory",
            "operation": {
                "op": "add_edge",
                "edge": {
                    "id": "edge-1",
                    "source": "chat-input-1",
                    "target": "memory-1",
                    "sourceHandle": "output",
                    "targetHandle": "input"
                }
            }
        },
        {
            "step": 6,
            "description": "Connect memory to LLM",
            "validation": "Check if edge was created between memory and LLM",
            "operation": {
                "op": "add_edge",
                "edge": {
                    "id": "edge-2",
                    "source": "memory-1",
                    "target": "llm-1",
                    "sourceHandle": "output",
                    "targetHandle": "input"
                }
            }
        },
        {
            "step": 7,
            "description": "Connect LLM to chat output",
            "validation": "Check if edge was created between LLM and output",
            "operation": {
                "op": "add_edge",
                "edge": {
                    "id": "edge-3",
                    "source": "llm-1",
                    "target": "chat-output-1",
                    "sourceHandle": "output",
                    "targetHandle": "input"
                }
            }
        }
    ]
}


async def simulate_pfu_workflow():
    """Simulate the complete PFU workflow."""
    print("🚀 Starting PFU (Partial Flow Update) Workflow")
    print("=" * 50)
    
    # Step 1: Generate plan (this would normally call the AI endpoint)
    print("📋 Step 1: AI Planning")
    print(f"   Generated plan with {len(EXAMPLE_PLAN['operations'])} operations")
    print(f"   Objective: {EXAMPLE_PLAN['objective']}")
    print()
    
    # Step 2: Execute plan step by step
    print("⚡ Step 2: Step-by-Step Execution")
    print("   (This simulates the new /api/v1/airelius/pfu/execute endpoint)")
    print()
    
    # Simulate the execution service
    from langflow.airelius.service import PFUService
    service = PFUService()
    
    # Initial empty flow
    initial_flow = {"nodes": [], "edges": []}
    
    # Execute the plan
    result = service.execute_plan_step_by_step(EXAMPLE_PLAN, initial_flow)
    
    # Display results
    summary = result["execution_summary"]
    print(f"   ✅ Execution completed!")
    print(f"   📊 Summary: {summary['successful_steps']}/{summary['total_steps']} steps successful")
    print()
    
    if summary['failed_steps']:
        print("   ❌ Failed steps:")
        for step in summary['failed_steps']:
            print(f"      Step {step['step']}: {step['error']}")
        print()
    
    # Step 3: Show final flow structure
    print("🎯 Step 3: Final Flow Structure")
    final_flow = result["final_flow_data"]
    print(f"   Nodes: {len(final_flow.get('nodes', []))}")
    print(f"   Edges: {len(final_flow.get('edges', []))}")
    print()
    
    # Step 4: Show what was accomplished
    print("🎉 What We Accomplished:")
    print("   • Built a complete chat flow incrementally")
    print("   • Each step was validated before proceeding")
    print("   • No need to regenerate the entire flow")
    print("   • Much faster than AI generating everything at once")
    print()
    
    print("🔗 Next Steps:")
    print("   1. Use /api/v1/airelius/pfu/plan to generate plans")
    print("   2. Use /api/v1/airelius/pfu/execute to run them step by step")
    print("   3. Monitor progress and handle failures gracefully")
    print("   4. Scale to complex flows with hundreds of components")


def show_api_usage():
    """Show how to use the API endpoints."""
    print("🌐 API Usage Examples")
    print("=" * 50)
    
    print("1. Generate a PFU Plan:")
    print("   POST /api/v1/airelius/pfu/plan")
    print("   {")
    print('     "prompt": "Add a chat flow with memory",')
    print('     "flow_id": "your-flow-id"')
    print("   }")
    print()
    
    print("2. Execute the Plan Step by Step:")
    print("   POST /api/v1/airelius/pfu/execute")
    print("   {")
    print('     "plan": { /* plan from step 1 */ },')
    print('     "flow_id": "your-flow-id",')
    print('     "max_steps": 10')
    print("   }")
    print()
    
    print("3. Index Backend Files for Context:")
    print("   POST /api/v1/airelius/pfu/index/files")
    print("   {")
    print('     "patterns": ["/src/backend/**/*.py"]')
    print("   }")
    print()


if __name__ == "__main__":
    print("🎯 Langflow PFU System Demo")
    print("=" * 50)
    print()
    
    # Run the simulation
    asyncio.run(simulate_pfu_workflow())
    
    print()
    show_api_usage()
    
    print("✨ That's it! The PFU system makes flow generation much faster and more reliable.")
