from __future__ import annotations

import os
from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from langflow.api.utils import CurrentActiveUser
from langflow.services.deps import get_settings_service
from langflow.logging import logger
from langflow.interface.components import get_and_cache_all_types_dict  # type: ignore
from langflow.airelius.service import PFUService
from langflow.services.database.models.flow.model import Flow
from langflow.services.deps import session_scope
from langflow.services.settings.service import SettingsService
from langflow.services.auth.utils import get_current_active_user


router = APIRouter(prefix="/airelius", tags=["Airelius"])


class PFUPlanRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to drive PFU plan")
    flow_id: str | None = Field(None, description="Optional flow to target")
    files: list[str] | None = Field(default=None, description="Optional server-side file paths to include")


class PFUPlanResponse(BaseModel):
    accepted: bool
    plan_id: str
    message: str | None = None
    composed_prompt: str | None = None
    llm_response: str | None = None
    operations: list[dict[str, Any]] | None = None
    plan: dict[str, Any] | None = None
    retrieved_snippets: list[dict[str, Any]] | None = None


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
    """Accept a PFU planning request and generate operations using GPT-5.
    
    This endpoint:
    1. Composes a prompt with system context and retrieved snippets
    2. Calls GPT-5 to generate PFU operations
    3. Returns the LLM response and parsed operations
    """
    try:
        settings_service = get_settings_service()
        # Build or fetch components catalog and index to local vector DB
        all_types = await get_and_cache_all_types_dict(settings_service=settings_service)
        service = PFUService()
        
        # Only reset if explicitly requested or if no components are indexed
        current_count = service.retriever.count()
        if current_count == 0:
            logger.info("[PFU] No components indexed, performing initial indexing...")
            service.index_components(all_types, reset=True)
        else:
            logger.info(f"[PFU] Found {current_count} indexed components, skipping re-indexing")

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
        composed_prompt = service.compose_prompt(request.prompt, flow_data, retrieved)
        logger.info("[PFU] Composed planning prompt:\n{}", composed_prompt)
        
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
            llm_response = await llm.ainvoke(composed_prompt)
            llm_content = llm_response.content
            logger.info("[PFU] LLM response received: {}", llm_content[:200] + "..." if len(llm_content) > 200 else llm_content)
        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
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
        
        return PFUPlanResponse(
            accepted=True,
            plan_id=f"pfu-plan-{hash(composed_prompt) % 10000}",
            message="PFU planning completed successfully",
            composed_prompt=composed_prompt,
            llm_response=llm_content,
            operations=plan_data.get("operations") if plan_data else None,
            plan=plan_data,
            retrieved_snippets=retrieved,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"PFU planning failed: {e}")
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
            service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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
        service = PFUService()
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




