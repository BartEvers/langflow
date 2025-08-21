from __future__ import annotations

import os
import asyncio
from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse

from langflow.api.utils import CurrentActiveUser
from langflow.services.deps import get_settings_service
from langflow.logging import logger
from langflow.interface.components import get_and_cache_all_types_dict  # type: ignore
from langflow.airelius.service import PFUService
from langflow.services.database.models.flow.model import Flow
from langflow.services.deps import session_scope
from langflow.services.settings.service import SettingsService
from langflow.services.auth.utils import get_current_active_user


# Global service instance to persist embeddings
_pfu_service = None
_indexing_done = False

def get_pfu_service():
    """Get or create the PFU service instance."""
    global _pfu_service, _indexing_done
    if _pfu_service is None:
        _pfu_service = PFUService()
        logger.info("[PFU] Created new PFU service instance")
    return _pfu_service

def check_and_index_components(service: PFUService, settings_service: SettingsService):
    """Check if components need indexing and do it if needed."""
    global _indexing_done
    if not _indexing_done:
        current_count = service.retriever.count()
        if current_count == 0:
            logger.info("[PFU] No components indexed, performing initial indexing...")
            # Build or fetch components catalog and index to local vector DB
            all_types = get_and_cache_all_types_dict(settings_service=settings_service)
            service.index_components(all_types, reset=True)
            _indexing_done = True
            logger.info("[PFU] Component indexing completed")
        else:
            logger.info(f"[PFU] Found {current_count} indexed components, skipping re-indexing")
            _indexing_done = True
    else:
        logger.info("[PFU] Components already indexed, skipping re-indexing")


router = APIRouter(prefix="/airelius", tags=["Airelius"])


class PFUPlanRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to drive PFU plan")
    flow_id: str | None = Field(None, description="Optional flow to target")
    files: list[str] | None = Field(default=None, description="Optional server-side file paths to include")
    available_templates: list[str] | None = Field(default=None, description="List of available component templates")


class PFUPlanResponse(BaseModel):
    accepted: bool
    plan_id: str
    message: str | None = None
    composed_prompt: str | None = None
    llm_response: str | None = None
    operations: list[dict[str, Any]] | None = None
    plan: dict[str, Any] | None = None
    retrieved_snippets: list[dict[str, Any]] | None = None
    debug_info: dict[str, Any] | None = None


class PFUExecuteRequest(BaseModel):
    plan: dict[str, Any] = Field(..., description="The PFU plan to execute")
    flow_id: str = Field(..., description="Flow ID to apply operations to")
    max_steps: int = Field(default=10, description="Maximum number of steps to execute")


class PFUExecuteResponse(BaseModel):
    success: bool
    message: str
    execution_summary: dict[str, Any] | None = None
    final_flow_data: dict[str, Any] | None = None


@router.post("/pfu/plan", response_model=PFUPlanResponse)
async def plan_pfu(
    request: PFUPlanRequest,
    current_user: CurrentActiveUser,
):
    """Accept a PFU planning request and generate operations using GPT-5."""
    try:
        settings_service = get_settings_service()
        service = get_pfu_service()  # Use global service
        
        # Check and index components if needed
        check_and_index_components(service, settings_service)

        # Optionally retrieve snippets relevant to the prompt (and flow signature later)
        retrieved = service.retrieve(request.prompt, k=8)
        logger.info(f"[PFU] Retrieved {len(retrieved)} snippets from vector DB")
        
        # Log what was retrieved for debugging
        for i, snippet in enumerate(retrieved[:3]):  # Log first 3 snippets
            logger.debug(f"[PFU] Snippet {i+1}: {snippet.get('text', '')[:100]}...")
        
        if not retrieved:
            logger.warning("[PFU] No snippets retrieved from vector DB - this might affect LLM performance")

        # Load flow data if provided to build a signature
        flow_data = None
        # We only have flow_id in request. Retrieve minimal data if possible
        # Avoid heavy joins; just fetch flow record
        # Using DbSession here for convenience
        if request.flow_id:
            from sqlmodel import select
            from uuid import UUID
            try:
                # Convert string flow_id to UUID
                flow_uuid = UUID(request.flow_id)
                async with session_scope() as session:
                    flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                    flow_data = flow.data if flow else None
            except ValueError as e:
                logger.warning(f"Invalid flow_id format: {request.flow_id}, error: {e}")
                flow_data = None

        # Compose a larger prompt by injecting user prompt and retrieved snippets
        print(f"\n{'='*80}")
        print(f"🎯 COMPOSING PROMPT FOR LLM")
        print(f"{'='*80}")
        print(f"📝 User Prompt: {request.prompt}")
        print(f"🔄 Flow ID: {request.flow_id}")
        print(f"📊 Retrieved {len(retrieved)} snippets")
        print(f"📋 Flow Data Available: {'Yes' if flow_data else 'No'}")
        print(f"{'='*80}\n")
        
        composed_prompt = service.compose_prompt(request.prompt, flow_data, retrieved)
        logger.info("[PFU] Composed planning prompt:\n{}", composed_prompt)
        
        print(f"\n{'='*80}")
        print(f"📤 SENDING PROMPT TO LLM")
        print(f"{'='*80}")
        print(f"🔑 OpenAI API Key: {'✅ Set' if openai_api_key else '❌ Missing'}")
        print(f"🤖 Model: gpt-4o")
        print(f"🌡️  Temperature: 0.1")
        print(f"📏 Max Tokens: 4000")
        print(f"{'='*80}\n")
        
        # Get OpenAI API key from environment or settings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        # Initialize GPT-5 model
        try:
            llm = ChatOpenAI(
                model="gpt-4o",  # Using GPT-4o as GPT-5 is not yet available
                api_key=openai_api_key,
                temperature=0.1,
                max_tokens=4000,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize OpenAI client: {str(e)}"
            )
        
        # Call the LLM with the composed prompt
        try:
            logger.info("[PFU] Calling LLM with composed prompt...")
            print(f"⏳ Calling LLM... (this may take a moment)")
            
            llm_response = await llm.ainvoke(composed_prompt)
            llm_content = llm_response.content
            
            print(f"\n{'='*80}")
            print(f"🤖 LLM RESPONSE RECEIVED")
            print(f"{'='*80}")
            print(llm_content)
            print(f"{'='*80}")
            print(f"📊 RESPONSE STATISTICS:")
            print(f"   Total length: {len(llm_content)} characters")
            print(f"   Total lines: {llm_content.count(chr(10)) + 1}")
            print(f"   Contains JSON: {'Yes' if '{' in llm_content and '}' in llm_content else 'No'}")
            print(f"{'='*80}\n")
            
            logger.info("[PFU] LLM response received: {}", llm_content[:200] + "..." if len(llm_content) > 200 else llm_content)
        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
            print(f"\n❌ LLM CALL FAILED: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to call LLM: {str(e)}"
            )
        
        # Log the LLM response for debugging
        logger.info(f"[PFU] LLM response length: {len(llm_content)} characters")
        logger.debug(f"[PFU] LLM response preview: {llm_content[:300]}...")
        
        # Use the service's improved plan parsing method
        plan_data = service.parse_plan_from_llm_response(llm_content)
        
        # Debug logging to see what we're sending
        logger.info(f"[PFU] Router: plan_data keys: {list(plan_data.keys()) if plan_data else 'None'}")
        logger.info(f"[PFU] Router: plan_data operations: {plan_data.get('operations', 'None') if plan_data else 'None'}")
        logger.info(f"[PFU] Router: plan_data type: {type(plan_data)}")
        logger.info(f"[PFU] Router: plan_data content: {plan_data}")
        
        # Create response with detailed prompt information
        response_data = {
            "accepted": True,
            "plan_id": f"pfu-plan-{hash(composed_prompt) % 10000}",
            "message": "PFU planning completed successfully",
            "composed_prompt": composed_prompt,
            "llm_response": llm_content,
            "operations": plan_data.get("operations") if plan_data else None,
            "plan": plan_data,
            "retrieved_snippets": retrieved,
            "debug_info": {
                "prompt_statistics": {
                    "total_length": len(composed_prompt),
                    "total_lines": composed_prompt.count(chr(10)) + 1,
                    "flow_data_included": flow_data is not None,
                    "snippets_included": len(retrieved) if retrieved else 0,
                    "flow_summary": {
                        "nodes_count": len(flow_data.get("nodes", [])) if flow_data else 0,
                        "edges_count": len(flow_data.get("edges", [])) if flow_data else 0,
                        "node_types": [node.get("type", "unknown") for node in flow_data.get("nodes", [])[:5]] if flow_data else []
                    } if flow_data else None
                },
                "llm_config": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                "retrieval_info": {
                    "query": request.prompt,
                    "k": 8,
                    "actual_retrieved": len(retrieved) if retrieved else 0
                }
            }
        }
        
        return PFUPlanResponse(**response_data)
    except Exception as e:  # noqa: BLE001
        logger.error(f"PFU planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/pfu/plan/step-by-step", response_model=dict[str, Any])
async def plan_pfu_step_by_step(
    request: PFUPlanRequest,
    current_user: CurrentActiveUser,
):
    """Execute PFU planning step by step using the dynamic step generation system."""
    try:
        settings_service = get_settings_service()
        service = get_pfu_service()
        
        # Check and index components if needed
        check_and_index_components(service, settings_service)

        # Retrieve relevant snippets
        retrieved = service.retrieve(request.prompt, k=8)
        logger.info(f"[PFU] Retrieved {len(retrieved)} snippets from vector DB")
        
        # Load flow data if provided
        flow_data = None
        if request.flow_id:
            from sqlmodel import select
            from uuid import UUID
            try:
                flow_uuid = UUID(request.flow_id)
                async with session_scope() as session:
                    flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                    flow_data = flow.data if flow else None
            except ValueError as e:
                logger.warning(f"Invalid flow_id format: {request.flow_id}, error: {e}")
                flow_data = None

        # Execute dynamic step-by-step PFU using the new method
        logger.info("[PFU] Starting dynamic step-by-step PFU execution...")
        result = service.execute_step_by_step_pfu(request.prompt, flow_data, retrieved, request.available_templates)
        
        return {
            "status": result["status"],
            "message": result["message"],
            "total_steps": result["execution_summary"]["total_steps"],
            "completed_steps": result["execution_summary"]["completed_steps"],
            "execution_summary": result["execution_summary"]
        }
        
    except Exception as e:  # noqa: BLE001
        logger.error(f"Direct PFU response generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/pfu/execute", response_model=PFUExecuteResponse)
async def execute_pfu_plan(
    request: PFUExecuteRequest,
    current_user: CurrentActiveUser,
):
    """Execute a PFU plan step by step with validation between steps.
    
    This endpoint:
    1. Takes a pre-generated PFU plan
    2. Executes operations incrementally (like Cursor)
    3. Validates each step before proceeding
    4. Returns execution summary and final flow data
    """
    try:
        # Validate the plan structure
        if not request.plan or "operations" not in request.plan:
            raise HTTPException(status_code=400, detail="Invalid plan: missing operations")
        
        # Load the target flow
        from sqlmodel import select
        from uuid import UUID
        try:
            flow_uuid = UUID(request.flow_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid flow_id format: {request.flow_id}")
        
        async with session_scope() as session:
            flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
            if not flow:
                raise HTTPException(status_code=404, detail="Flow not found")
            
            # Check if user has access to this flow
            if flow.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied to this flow")
            
            # Execute the plan step by step
            service = get_pfu_service()
            result = service.execute_plan_step_by_step(
                request.plan, 
                flow.data, 
                max_steps=request.max_steps
            )
            
            # Update the flow with the final data
            flow.data = result["final_flow_data"]
            flow.updated_at = datetime.now(timezone.utc)
            
            # Save the updated flow
            session.add(flow)
            await session.commit()
            await session.refresh(flow)
            
            logger.info(f"[PFU] Successfully executed plan for flow {request.flow_id}")
            
            return PFUExecuteResponse(
                success=True,
                message="PFU plan executed successfully",
                execution_summary=result["execution_summary"],
                final_flow_data=result["final_flow_data"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PFU execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e





class PFUIndexFilesRequest(BaseModel):
    patterns: list[str] = Field(..., description="List of absolute paths or glob patterns to index")
    reset: bool = Field(default=False, description="Whether to reset the collection before indexing")
    chunk_size: int = 2000
    overlap: int = 200


class PFUIndexFilesResponse(BaseModel):
    files: int
    chunks: int


@router.post("/pfu/index/files", response_model=PFUIndexFilesResponse)
async def index_files(
    request: PFUIndexFilesRequest,
    current_user: CurrentActiveUser,
):
    try:
        service = get_pfu_service()
        stats = service.index_files(
            request.patterns, reset=request.reset, chunk_size=request.chunk_size, overlap=request.overlap
        )
        return PFUIndexFilesResponse(**stats)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


# -------- Introspection endpoints --------

class PFUIndexCountResponse(BaseModel):
    count: int


@router.get("/pfu/index/count", response_model=PFUIndexCountResponse)
async def index_count(current_user: CurrentActiveUser):
    try:
        service = get_pfu_service()
        return PFUIndexCountResponse(count=service.retriever.count())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


class PFUIndexSampleResponse(BaseModel):
    items: list[dict]


@router.get("/pfu/index/sample", response_model=PFUIndexSampleResponse)
async def index_sample(current_user: CurrentActiveUser, n: int = 5):
    try:
        service = get_pfu_service()
        return PFUIndexSampleResponse(items=service.retriever.sample(n))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/index/debug")
async def debug_retriever(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
) -> dict[str, Any]:
    """Debug the retriever state."""
    try:
        service = get_pfu_service()
        debug_info = service.retriever.debug_info()
        return {
            "retriever_state": debug_info,
            "current_count": service.retriever.count(),
            "sample_data": service.retriever.sample(2)
        }
    except Exception as e:
        logger.error(f"[PFU] Debug failed: {e}")
        return {"error": str(e)}

@router.post("/index/force-simple")
async def force_simple_mode(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
) -> dict[str, Any]:
    """Force the retriever to use simple mode."""
    try:
        service = get_pfu_service()
        service.retriever.force_simple_mode()
        return {
            "message": "Forced simple mode",
            "new_state": service.retriever.debug_info()
        }
    except Exception as e:
        logger.error(f"[PFU] Force simple mode failed: {e}")
        return {"error": str(e)}


@router.post("/index/reload")
async def reload_documents(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
) -> dict[str, Any]:
    """Force reload documents from the simple store file."""
    try:
        service = get_pfu_service()
        reloaded_count = service.retriever.reload_documents()
        return {
            "message": f"Reloaded {reloaded_count} documents",
            "new_state": service.retriever.debug_info()
        }
    except Exception as e:
        logger.error(f"[PFU] Reload documents failed: {e}")
        return {"error": str(e)}


@router.post("/index/test-query")
async def test_query(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
    query: str = "test query",
    k: int = 3
) -> dict[str, Any]:
    """Test a query to debug the retrieval process."""
    try:
        service = get_pfu_service()
        logger.info(f"[PFU] Testing query: '{query}' with k={k}")
        
        # Get current state
        current_count = service.retriever.count()
        logger.info(f"[PFU] Current retriever count: {current_count}")
        
        # Try to retrieve
        retrieved = service.retrieve(query, k=k)
        logger.info(f"[PFU] Retrieved {len(retrieved)} results")
        
        return {
            "query": query,
            "k": k,
            "current_count": current_count,
            "retrieved_count": len(retrieved),
            "retrieved": retrieved,
            "retriever_state": service.retriever.debug_info()
        }
    except Exception as e:
        logger.error(f"[PFU] Test query failed: {e}")
        return {"error": str(e)}


@router.post("/index/test-snippets")
async def test_snippets_retrieval(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
    prompt: str = "test prompt",
    k: int = 5,
    include_samples: bool = True,
    include_debug: bool = True
) -> dict[str, Any]:
    """Test snippets retrieval based on a prompt with comprehensive debugging information."""
    try:
        service = get_pfu_service()
        logger.info(f"[PFU] Testing snippets retrieval for prompt: '{prompt[:100]}...' with k={k}")
        
        # Get current state
        current_count = service.retriever.count()
        logger.info(f"[PFU] Current retriever count: {current_count}")
        
        if current_count == 0:
            return {
                "error": "No documents indexed in the vector database",
                "current_count": current_count,
                "suggestion": "Run /index/reload to index components first"
            }
        
        # Try to retrieve snippets
        retrieved = service.retrieve(prompt, k=k)
        logger.info(f"[PFU] Retrieved {len(retrieved)} snippets")
        
        # Build response
        response = {
            "prompt": prompt,
            "k": k,
            "current_count": current_count,
            "retrieved_count": len(retrieved),
            "retrieved_snippets": retrieved,
            "retrieval_summary": {
                "total_documents": current_count,
                "retrieved_percentage": round((len(retrieved) / current_count) * 100, 2) if current_count > 0 else 0,
                "average_snippet_length": round(sum(len(snippet.get("text", "")) for snippet in retrieved) / len(retrieved), 2) if retrieved else 0
            }
        }
        
        # Add sample documents if requested
        if include_samples and current_count > 0:
            sample_docs = service.retriever.sample(min(3, current_count))
            response["sample_documents"] = sample_docs
        
        # Add debug information if requested
        if include_debug:
            response["retriever_debug_info"] = service.retriever.debug_info()
        
        # Add metadata analysis
        if retrieved:
            meta_keys = set()
            for snippet in retrieved:
                if snippet.get("meta"):
                    meta_keys.update(snippet["meta"].keys())
            response["metadata_analysis"] = {
                "unique_metadata_keys": list(meta_keys),
                "snippet_types": list(set(snippet.get("meta", {}).get("type", "unknown") for snippet in retrieved))
            }
        
        return response
        
    except Exception as e:
        logger.error(f"[PFU] Snippets retrieval test failed: {e}")
        return {"error": str(e), "traceback": str(e.__traceback__)}


@router.post("/index/compare-prompts")
async def compare_prompts_retrieval(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
    prompts: list[str] = ["test prompt", "component search", "flow creation"],
    k: int = 3
) -> dict[str, Any]:
    """Compare retrieval performance across multiple prompts to test vector search quality."""
    try:
        service = get_pfu_service()
        current_count = service.retriever.count()
        
        if current_count == 0:
            return {
                "error": "No documents indexed in the vector database",
                "suggestion": "Run /index/reload to index components first"
            }
        
        results = {}
        total_retrieved = 0
        
        for prompt in prompts:
            logger.info(f"[PFU] Testing prompt: '{prompt[:50]}...'")
            retrieved = service.retrieve(prompt, k=k)
            results[prompt] = {
                "retrieved_count": len(retrieved),
                "snippets": retrieved,
                "snippet_previews": [
                    {
                        "id": snippet.get("id"),
                        "text_preview": snippet.get("text", "")[:150] + "..." if len(snippet.get("text", "")) > 150 else snippet.get("text", ""),
                        "meta_type": snippet.get("meta", {}).get("type", "unknown"),
                        "meta_name": snippet.get("meta", {}).get("name", "unknown")
                    }
                    for snippet in retrieved
                ]
            }
            total_retrieved += len(retrieved)
        
        # Calculate overlap between different prompts
        overlap_analysis = {}
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts[i+1:], i+1):
                ids1 = set(snippet.get("id") for snippet in results[prompt1]["snippets"])
                ids2 = set(snippet.get("id") for snippet in results[prompt2]["snippets"])
                overlap = len(ids1.intersection(ids2))
                overlap_analysis[f"{prompt1[:20]}... vs {prompt2[:20]}..."] = {
                    "overlap_count": overlap,
                    "overlap_percentage": round((overlap / k) * 100, 2) if k > 0 else 0
                }
        
        return {
            "prompts_tested": prompts,
            "k": k,
            "total_documents_indexed": current_count,
            "total_snippets_retrieved": total_retrieved,
            "results_by_prompt": results,
            "overlap_analysis": overlap_analysis,
            "retrieval_quality_metrics": {
                "average_retrieved_per_prompt": round(total_retrieved / len(prompts), 2),
                "prompts_with_results": len([p for p in prompts if results[p]["retrieved_count"] > 0]),
                "total_unique_snippets": len(set(
                    snippet.get("id") 
                    for prompt_results in results.values() 
                    for snippet in prompt_results["snippets"]
                ))
            }
        }
        
    except Exception as e:
        logger.error(f"[PFU] Prompt comparison failed: {e}")
        return {"error": str(e)}


@router.post("/index/test-prompt-composition")
async def test_prompt_composition(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service),
    prompt: str = "test prompt",
    flow_id: str | None = None,
    k: int = 8
) -> dict[str, Any]:
    """Test prompt composition without calling the LLM - shows exactly what would be sent."""
    try:
        service = get_pfu_service()
        
        # Check if components are indexed
        current_count = service.retriever.count()
        if current_count == 0:
            return {
                "error": "No documents indexed in the vector database",
                "suggestion": "Run /index/reload to index components first"
            }
        
        # Retrieve snippets
        retrieved = service.retrieve(prompt, k=k)
        
        # Load flow data if provided
        flow_data = None
        if flow_id:
            from sqlmodel import select
            from uuid import UUID
            try:
                flow_uuid = UUID(flow_id)
                async with session_scope() as session:
                    flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                    flow_data = flow.data if flow else None
            except ValueError as e:
                return {"error": f"Invalid flow_id format: {flow_id}"}
        
        # Compose the prompt (this will show the CLI output in server logs)
        composed_prompt = service.compose_prompt(prompt, flow_data, retrieved)
        
        # Return detailed information about the prompt composition
        return {
            "test_info": {
                "prompt": prompt,
                "flow_id": flow_id,
                "k": k,
                "flow_data_available": flow_data is not None
            },
            "composed_prompt": composed_prompt,
            "prompt_analysis": {
                "total_length": len(composed_prompt),
                "total_lines": composed_prompt.count(chr(10)) + 1,
                "sections": {
                    "pfu_kernel": "PFU_KERNEL" in composed_prompt,
                    "components_section": "RELEVANT COMPONENTS FOR THIS REQUEST:" in composed_prompt,
                    "flow_section": "CURRENT FLOW STRUCTURE" in composed_prompt,
                    "user_request": "USER REQUEST:" in composed_prompt,
                    "planning_instructions": "PLANNING INSTRUCTIONS:" in composed_prompt
                },
                "content_breakdown": {
                    "pfu_kernel_length": composed_prompt.find("RELEVANT COMPONENTS FOR THIS REQUEST:") if "RELEVANT COMPONENTS FOR THIS REQUEST:" in composed_prompt else -1,
                    "components_length": composed_prompt.find("CURRENT FLOW STRUCTURE") - composed_prompt.find("RELEVANT COMPONENTS FOR THIS REQUEST:") if "RELEVANT COMPONENTS FOR THIS REQUEST:" in composed_prompt and "CURRENT FLOW STRUCTURE" in composed_prompt else -1,
                    "flow_length": composed_prompt.find("USER REQUEST:") - composed_prompt.find("CURRENT FLOW STRUCTURE") if "CURRENT FLOW STRUCTURE" in composed_prompt and "USER REQUEST:" in composed_prompt else -1,
                    "instructions_length": len(composed_prompt) - composed_prompt.find("PLANNING INSTRUCTIONS:") if "PLANNING INSTRUCTIONS:" in composed_prompt else -1
                }
            },
            "retrieval_info": {
                "total_documents": current_count,
                "retrieved_count": len(retrieved),
                "snippets_preview": [
                    {
                        "id": snippet.get("id"),
                        "text_preview": snippet.get("text", "")[:200] + "..." if len(snippet.get("text", "")) > 200 else snippet.get("text", ""),
                        "meta_type": snippet.get("meta", {}).get("type", "unknown")
                    }
                    for snippet in retrieved[:3]  # Show first 3 snippets
                ]
            },
            "flow_info": {
                "flow_data_included": flow_data is not None,
                "flow_summary": {
                    "nodes_count": len(flow_data.get("nodes", [])) if flow_data else 0,
                    "edges_count": len(flow_data.get("edges", [])) if flow_data else 0,
                    "node_types": [node.get("type", "unknown") for node in flow_data.get("nodes", [])[:5]] if flow_data else []
                } if flow_data else None
            }
        }
        
    except Exception as e:
        logger.error(f"[PFU] Prompt composition test failed: {e}")
        return {"error": str(e)}


@router.get("/index/status")
async def get_index_status(
    current_user: CurrentActiveUser,
    settings_service: SettingsService = Depends(get_settings_service)
) -> dict[str, Any]:
    """Get the current status of the vector database index."""
    try:
        service = get_pfu_service()
        
        # Get basic counts
        current_count = service.retriever.count()
        
        # Get debug info
        debug_info = service.retriever.debug_info()
        
        # Get sample documents
        sample_docs = service.retriever.sample(5) if current_count > 0 else []
        
        return {
            "index_status": {
                "total_documents": current_count,
                "index_ready": current_count > 0,
                "mode": debug_info.get("mode", "unknown"),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "retriever_info": debug_info,
            "sample_documents": sample_docs,
            "suggestions": {
                "reload_if_empty": current_count == 0,
                "mode_switch": "Consider switching to simple mode if ChromaDB has issues" if debug_info.get("mode") == "chroma" else None
            }
        }
        
    except Exception as e:
        logger.error(f"[PFU] Index status check failed: {e}")
        return {"error": str(e)}


from fastapi.responses import StreamingResponse
import asyncio
import json

# ... existing code ...

@router.post("/pfu/plan/step-by-step/stream")
async def plan_pfu_step_by_step_stream(
    request: PFUPlanRequest,
    current_user: CurrentActiveUser,
):
    """Execute PFU planning step by step with dynamic step generation and real-time streaming."""
    
    async def generate_stream():
        try:
            settings_service = get_settings_service()
            service = get_pfu_service()
            
            # Check and index components if needed
            check_and_index_components(service, settings_service)
            
            # Retrieve relevant snippets
            retrieved = service.retrieve(request.prompt, k=8)
            logger.info(f"[PFU] Retrieved {len(retrieved)} snippets from vector DB")
            
            # Load flow data if provided
            flow_data = None
            if request.flow_id:
                from sqlmodel import select
                from uuid import UUID
                try:
                    flow_uuid = UUID(request.flow_id)
                    async with session_scope() as session:
                        flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                        flow_data = flow.data if flow else None
                except ValueError as e:
                    logger.warning(f"Invalid flow_id format: {request.flow_id}, error: {e}")
                    flow_data = None

            # Execute dynamic step-by-step PFU with streaming
            logger.info("[PFU] Starting dynamic streaming step-by-step PFU execution...")
            
            # CLI logging for user prompt
            print(f"\n{'='*80}")
            print(f"🎯 PFU PLANNING REQUEST")
            print(f"{'='*80}")
            print(f"📝 User Prompt: {request.prompt}")
            print(f"🔄 Flow ID: {request.flow_id}")
            print(f"📊 Retrieved {len(retrieved)} relevant snippets")
            print(f"{'='*80}\n")
            
            # Step 1: Analyze request complexity and generate custom step plan
            print(f"🔄 Step 1: Analyzing request complexity...")
            yield f"data: {json.dumps({'step': 1, 'name': 'complexity_analysis', 'status': 'starting', 'message': 'Analyzing request complexity...'})}\n\n"
            
            # Use the service to analyze complexity and get custom steps
            complexity_analysis = service._analyze_request_complexity(request.prompt, flow_data, retrieved)
            custom_steps = complexity_analysis["required_steps"]
            total_steps = len(custom_steps)
            
            print(f"✅ Complexity analysis completed: {total_steps} steps needed")
            print(f"   Steps: {[step['name'] for step in custom_steps]}")
            yield f"data: {json.dumps({'step': 1, 'name': 'complexity_analysis', 'status': 'completed', 'result': complexity_analysis, 'message': f'Complexity analysis completed: {total_steps} steps needed'})}\n\n"
            
            # Execute the custom steps dynamically
            executed_steps = {}
            previous_steps = {}
            
            for i, step_config in enumerate(custom_steps):
                step_num = i + 2  # Start from 2 since complexity analysis was step 1
                step_name = step_config["name"]
                step_description = step_config["description"]
                step_type = step_config["type"]
                
                print(f"🔄 Step {step_num}/{total_steps + 1}: {step_name} - Starting...")
                yield f"data: {json.dumps({'step': step_num, 'name': step_name, 'status': 'starting', 'message': f'Starting {step_name}...'})}\n\n"
                
                # Execute this step
                step_result = await service.execute_single_step(step_name, request.prompt, flow_data, retrieved, previous_steps, request.available_templates)
                print(f"✅ Step {step_num}/{total_steps + 1}: {step_name} - Completed")
                print(f"   Result: {str(step_result)[:200]}...")
                
                # Stream the LLM response character by character
                if step_result and isinstance(step_result, dict) and 'response' in step_result:
                    response_text = step_result['response']
                    print(f" Streaming response ({len(response_text)} characters)...")
                    for char in response_text:
                        yield f"data: {json.dumps({'step': step_num, 'content': char, 'type': 'streaming'})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for streaming effect
                    print(f"✅ Response streaming completed")
                
                yield f"data: {json.dumps({'step': step_num, 'name': step_name, 'status': 'completed', 'result': step_result, 'message': f'{step_name} completed'})}\n\n"
                
                # Store the result for next steps
                executed_steps[step_name] = step_result
                previous_steps[step_name] = step_result
                
                print(f"Step {step_num}/{total_steps + 1} completed: {step_name}")
            
            # Final completion
            final_plan = []
            if executed_steps:
                # Try to extract operations from the last step
                last_step = list(executed_steps.values())[-1]
                final_plan = last_step.get("operations", [])
                if not final_plan and isinstance(last_step, dict):
                    # Try to find operations in the response
                    response = last_step.get("response", "")
                    if "operations" in response.lower():
                        # Try to extract JSON operations from the response
                        try:
                            import re
                            json_match = re.search(r'\{.*"operations".*\}', response, re.DOTALL)
                            if json_match:
                                parsed = json.loads(json_match.group(0))
                                final_plan = parsed.get("operations", [])
                        except:
                            pass
            
            print(f"\n🎯 ALL STEPS COMPLETED!")
            print(f"📋 Final Plan Operations: {len(final_plan)} operations generated")
            print(f"📊 Total Steps Executed: {total_steps + 1} (including complexity analysis)")
            print(f"{'='*80}")
            
            yield f"data: {json.dumps({'status': 'completed', 'total_steps': total_steps + 1, 'final_plan': final_plan, 'message': f'All {total_steps + 1} steps completed successfully!'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming PFU planning failed: {e}")
            yield f"data: {json.dumps({'error': str(e), 'message': 'An error occurred during execution'})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.post("/pfu/chat/stream")
async def chat_pfu_stream(
    request: PFUPlanRequest,
    current_user: CurrentActiveUser,
):
    """Streaming chat with PFU service."""
    
    async def generate_stream():
        try:
            settings_service = get_settings_service()
            service = get_pfu_service()
            
            # Check and index components if needed
            check_and_index_components(service, settings_service)
            
            # Retrieve relevant snippets
            retrieved = service.retrieve(request.prompt, k=8)
            logger.info(f"[PFU] Retrieved {len(retrieved)} snippets from vector DB")
            
            # Load flow data if provided
            flow_data = None
            if request.flow_id:
                from sqlmodel import select
                from uuid import UUID
                try:
                    flow_uuid = UUID(request.flow_id)
                    async with session_scope() as session:
                        flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                        flow_data = flow.data if flow else None
                except ValueError as e:
                    logger.warning(f"Invalid flow_id format: {request.flow_id}, error: {e}")
                    flow_data = None

            # Generate streaming response
            logger.info("[PFU] Starting streaming chat...")
            
            # CLI logging for chat request
            print(f"\n{'='*80}")
            print(f"💬 PFU CHAT REQUEST")
            print(f"{'='*80}")
            print(f"📝 User Question: {request.prompt}")
            print(f"🔄 Flow ID: {request.flow_id}")
            print(f"📊 Retrieved {len(retrieved)} relevant snippets")
            print(f"{'='*80}\n")
            
            # Use the service to generate a chat response
            response = await service.chat(request.prompt, flow_data, retrieved)
            
            # Stream the response character by character for a ChatGPT-like experience
            if response and isinstance(response, str):
                print(f"🤖 AI Response Generated ({len(response)} characters)")
                print(f"📝 Response Preview: {response[:200]}...")
                print(f"🔄 Starting to stream response...")
                
                for char in response:
                    yield f"data: {json.dumps({'content': char})}\n\n"
                    await asyncio.sleep(0.015)  # Balanced delay for responsive but realistic streaming
                
                print(f"✅ Response streaming completed")
            else:
                # Handle empty or invalid responses
                print(f"❌ No response generated")
                yield f"data: {json.dumps({'content': 'I apologize, but I could not generate a response. Please try again.'})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            print(f"🏁 Chat session completed")
            print(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield f"data: {json.dumps({'error': str(e), 'message': 'An error occurred during chat'})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")
