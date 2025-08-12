# Airelius Snippets Retrieval Testing

This document describes the new endpoints created for testing snippets retrieval in the Airelius vector database system.

## New Endpoints

### 1. Test Snippets Retrieval
**POST** `/api/v1/airelius/index/test-snippets`

Test snippets retrieval based on a prompt with comprehensive debugging information.

**Parameters:**
- `prompt` (string): The prompt to test retrieval with
- `k` (int, default: 5): Number of snippets to retrieve
- `include_samples` (bool, default: true): Include sample documents in response
- `include_debug` (bool, default: true): Include debug information

**Example Request:**
```bash
curl -X POST "http://localhost:7860/api/v1/airelius/index/test-snippets" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to create a chat component with memory",
    "k": 5,
    "include_samples": true,
    "include_debug": true
  }'
```

**Response:**
```json
{
  "prompt": "How to create a chat component with memory",
  "k": 5,
  "current_count": 1234,
  "retrieved_count": 5,
  "retrieved_snippets": [...],
  "retrieval_summary": {
    "total_documents": 1234,
    "retrieved_percentage": 0.41,
    "average_snippet_length": 245.6
  },
  "sample_documents": [...],
  "retriever_debug_info": {...},
  "metadata_analysis": {
    "unique_metadata_keys": ["type", "name", "description"],
    "snippet_types": ["component", "tool"]
  }
}
```

### 2. Compare Prompts Retrieval
**POST** `/api/v1/airelius/index/compare-prompts`

Compare retrieval performance across multiple prompts to test vector search quality.

**Parameters:**
- `prompts` (list[string]): List of prompts to compare
- `k` (int, default: 3): Number of snippets to retrieve per prompt

**Example Request:**
```bash
curl -X POST "http://localhost:7860/api/v1/airelius/index/compare-prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "component creation",
      "flow building",
      "API integration"
    ],
    "k": 3
  }'
```

**Response:**
```json
{
  "prompts_tested": ["component creation", "flow building", "API integration"],
  "k": 3,
  "total_documents_indexed": 1234,
  "total_snippets_retrieved": 9,
  "results_by_prompt": {...},
  "overlap_analysis": {
    "component creation... vs flow building...": {
      "overlap_count": 1,
      "overlap_percentage": 33.33
    }
  },
  "retrieval_quality_metrics": {
    "average_retrieved_per_prompt": 3.0,
    "prompts_with_results": 3,
    "total_unique_snippets": 8
  }
}
```

### 3. Index Status
**GET** `/api/v1/airelius/index/status`

Get the current status of the vector database index.

**Example Request:**
```bash
curl "http://localhost:7860/api/v1/airelius/index/status"
```

**Response:**
```json
{
  "index_status": {
    "total_documents": 1234,
    "index_ready": true,
    "mode": "simple",
    "last_updated": "2024-01-15T10:30:00Z"
  },
  "retriever_info": {...},
  "sample_documents": [...],
  "suggestions": {
    "reload_if_empty": false,
    "mode_switch": null
  }
}
```

## Testing Script

A comprehensive testing script is provided in `test_snippets_retrieval.py` that demonstrates how to use all these endpoints.

**Usage:**
```bash
cd examples
python test_snippets_retrieval.py
```

**Features:**
- Tests index status
- Tests snippets retrieval with various prompts
- Compares retrieval across multiple prompts
- Tests specific use cases common in Langflow
- Provides detailed output and analysis

## Use Cases

These endpoints are useful for:

1. **Debugging Vector Search**: Understand why certain prompts return specific results
2. **Quality Assurance**: Test if the vector database is working correctly
3. **Performance Testing**: Compare retrieval performance across different types of queries
4. **Development**: Verify that new components are properly indexed
5. **Troubleshooting**: Diagnose issues with the retrieval system

## Prerequisites

Before using these endpoints:

1. Ensure Langflow is running
2. Make sure components are indexed (run `/index/reload` if needed)
3. The vector database should contain documents

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Error description",
  "suggestion": "What to do next"
}
```

Common errors:
- **No documents indexed**: Run `/index/reload` first
- **Vector database issues**: Check the retriever mode and configuration
- **Authentication errors**: Ensure proper user authentication

## Configuration

The endpoints respect the following environment variables:
- `AIRELIUS_RETRIEVER_MODE`: Set to "simple" or "chroma"
- `AIRELIUS_SIMPLE_METHOD`: Set to "st" (sentence-transformers) or "tfidf"
- `AIRELIUS_EMBED_MODEL`: Custom embedding model name

## Troubleshooting

If you encounter issues:

1. Check the index status first
2. Verify that components are properly indexed
3. Check the retriever debug information
4. Consider switching retriever modes if ChromaDB has issues
5. Review the logs for detailed error information
