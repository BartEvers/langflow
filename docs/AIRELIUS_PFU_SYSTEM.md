# Airelius PFU System: Partial Flow Updates

## Overview

The Airelius PFU (Partial Flow Update) system is Langflow's solution to the performance problem of AI-generated flows. Instead of generating entire flows from scratch (which can take minutes for complex flows), PFU uses a **multi-step planning and incremental execution approach** similar to how Cursor handles code changes.

## The Problem

Traditional AI flow generation:
- ❌ Generates entire flows at once
- ❌ Can take 2-5 minutes for complex flows (2000-3000 lines)
- ❌ Wastes time regenerating unchanged parts
- ❌ No progress tracking or validation

## The Solution

PFU System:
- ✅ **Planning Phase**: AI creates a strategic plan with step-by-step operations
- ✅ **Execution Phase**: Operations are applied incrementally with validation
- ✅ **Progress Tracking**: Real-time feedback on each step
- ✅ **Rollback Capability**: Failed steps don't break the entire flow
- ✅ **Performance**: 10-50x faster than full regeneration
- ✅ **Individual Operation Control**: Execute single operations or entire plans

## Architecture

```
User Request → AI Planning → Step-by-Step Execution → Validated Flow
     ↓              ↓              ↓                    ↓
  "Add chat" → Strategic Plan → Add Node 1 → Final Flow
              → Operations    → Add Node 2 → with all
              → Validation    → Connect... → components
```

## API Endpoints

### 1. Generate PFU Plan
```http
POST /api/v1/airelius/pfu/plan
```

**Request:**
```json
{
  "prompt": "Add a chat flow with memory and OpenAI integration",
  "flow_id": "optional-flow-id-for-context",
  "files": ["optional-server-file-paths"]
}
```

**Response:**
```json
{
  "accepted": true,
  "plan_id": "pfu-plan-1234",
  "plan": {
    "objective": "Add chat flow with memory",
    "operations": [
      {
        "step": 1,
        "description": "Add chat input node",
        "validation": "Check if node was added",
        "operation": {
          "op": "add_node",
          "node": { /* node data */ }
        }
      }
    ]
  },
  "llm_response": "AI's reasoning and planning process"
}
```

### 2. Execute PFU Plan
```http
POST /api/v1/airelius/pfu/execute
```

**Request:**
```json
{
  "plan": { /* plan from step 1 */ },
  "flow_id": "target-flow-id",
  "max_steps": 10
}
```

**Response:**
```json
{
  "success": true,
  "message": "PFU plan executed successfully",
  "execution_summary": {
    "total_steps": 7,
    "successful_steps": 7,
    "failed_steps": 0,
    "executed_steps": [ /* step details */ ],
    "failed_steps": []
  },
  "final_flow_data": { /* updated flow */ }
}
```

### 3. Apply Single Operation
```http
POST /api/v1/flows/{flow_id}/ops
```

**Request:**
```json
{
  "operations": [
    {
      "op": "add_node",
      "node": { /* node data */ }
    }
  ]
}
```

**Response:** Returns the updated `FlowRead` object with the new flow data.

**Note:** This endpoint leverages Langflow's existing flow operations system, which provides optimistic concurrency control and built-in validation.

### 4. Index Backend Files
```http
POST /api/v1/airelius/pfu/index/files
```

**Request:**
```json
{
  "patterns": ["/src/backend/**/*.py"],
  "reset": false,
  "chunk_size": 2000,
  "overlap": 200
}
```

## Usage Workflow

### Step 1: Generate Plan
```python
import requests

# Generate a plan for adding chat functionality
response = requests.post("http://localhost:7860/api/v1/airelius/pfu/plan", json={
    "prompt": "Add a chat flow with memory and OpenAI integration",
    "flow_id": "your-flow-id"
})

plan = response.json()["plan"]
print(f"Generated plan with {len(plan['operations'])} operations")
```

### Step 2: Execute Plan (Option A - Full Plan)
```python
# Execute the entire plan step by step
response = requests.post("http://localhost:7860/api/v1/airelius/pfu/execute", json={
    "plan": plan,
    "flow_id": "your-flow-id",
    "max_steps": 10
})

result = response.json()
if result["success"]:
    summary = result["execution_summary"]
    print(f"✅ Executed {summary['successful_steps']}/{summary['total_steps']} steps")
else:
    print("❌ Execution failed")
```

### Step 2: Execute Plan (Option B - Individual Operations)
```python
# Execute operations one by one for more control
for operation in plan["operations"]:
    response = requests.post(f"http://localhost:7860/api/v1/flows/your-flow-id/ops", json={
        "operations": [operation["operation"]]
    })
    
    if response.status_code == 200:
        print(f"✅ Step {operation['step']} completed")
    else:
        print(f"❌ Step {operation['step']} failed")
```

### Step 3: Index Backend Files for Context
```python
# Index backend files to give AI better context
response = requests.post("http://localhost:7860/api/v1/airelius/pfu/index/files", json={
    "patterns": ["/src/backend/**/*.py"]
})

print(f"Indexed {response.json()['files']} files")
```

## Frontend Integration

The PFU system includes a React component (`AireliusChat`) that provides:

- **Chat Interface**: Natural language interaction with the AI
- **Plan Visualization**: Shows the AI's planning process and operations
- **Individual Execution**: Execute single operations or entire plans
- **Progress Tracking**: Real-time feedback on execution
- **Error Handling**: Graceful handling of failures

### Key Features:
- **Hover-Expanded Chat**: Minimal UI footprint when not in use
- **Plan Display**: Shows AI reasoning, objectives, and operations
- **Operation Controls**: Individual execute buttons for each step
- **Execution Results**: Detailed feedback on success/failure
- **File Indexing**: Easy backend file indexing for context

## Key Features

### 1. **Strategic Planning**
- AI analyzes the current flow structure
- Creates logical operation sequences
- Considers dependencies and validation
- Plans for incremental execution

### 2. **Incremental Execution**
- Operations applied one at a time
- Validation between each step
- Rollback capability for failed steps
- Progress tracking and reporting

### 3. **Flexible Control**
- Execute entire plans or individual operations using existing Langflow endpoints
- Real-time feedback on each step
- Detailed error reporting
- Support for all flow operation types

### 4. **Performance Optimization**
- No full flow regeneration
- Incremental updates only
- Parallel operation support (future)
- Smart caching and validation

### 5. **Native Integration**
- Leverages Langflow's existing `/api/v1/flows/{flow_id}/ops` endpoint
- Built-in optimistic concurrency control
- Standard flow validation and error handling
- Consistent with Langflow's architecture

## Operation Types Supported

The PFU system supports all standard flow operations:

- **`add_node`**: Add new component nodes
- **`update_node`**: Modify existing node properties
- **`remove_node`**: Remove nodes (and connected edges)
- **`add_edge`**: Connect nodes together
- **`remove_edge`**: Remove connections

## Error Handling

The system provides comprehensive error handling:

- **Plan Parsing**: Graceful fallback when LLM responses are malformed
- **Operation Validation**: Checks before applying each operation
- **Rollback Support**: Failed operations don't break the flow
- **Detailed Logging**: Comprehensive logging for debugging

## Testing

Use the provided test script to verify the system:

```bash
python examples/test_pfu_system.py
```

Make sure to:
1. Set your API key in the script
2. Have Langflow running
3. Adjust the base URL if needed

## Future Enhancements

- **Parallel Execution**: Execute independent operations simultaneously
- **Smart Rollback**: Automatic rollback of failed operation sequences
- **Flow Templates**: Pre-built operation patterns for common tasks
- **Performance Metrics**: Detailed timing and performance analysis
- **Advanced Validation**: Custom validation rules for specific components

## Troubleshooting

### Common Issues:

1. **Plan Parsing Fails**
   - Check LLM response format
   - Verify JSON structure
   - Check for malformed responses

2. **Operations Fail**
   - Verify flow ID exists
   - Check user permissions
   - Validate operation format

3. **Performance Issues**
   - Reduce max_steps parameter
   - Check operation complexity
   - Monitor LLM response times

### Debug Mode:
Enable detailed logging by setting the log level to DEBUG in your Langflow configuration.

## Conclusion

The Airelius PFU system transforms how AI interacts with Langflow flows, providing:

- **10-50x performance improvement** over full regeneration
- **Incremental, controlled updates** with validation
- **Professional-grade error handling** and rollback
- **Intuitive frontend interface** for easy interaction
- **Native Langflow integration** without duplicate endpoints
- **Leverages existing infrastructure** for better maintainability

This system makes AI-powered flow modification practical for production use, similar to how Cursor revolutionized AI-assisted coding.
