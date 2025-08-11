# Router for base api
from fastapi import APIRouter

from langflow.api.v1 import (
    api_key_router,
    chat_router,
    endpoints_router,
    files_router,
    flows_router,
    folders_router,
    login_router,
    mcp_projects_router,
    mcp_router,
    monitor_router,
    projects_router,
    starter_projects_router,
    store_router,
    users_router,
    validate_router,
    variables_router,
)
from langflow.api.v1.voice_mode import router as voice_mode_router
from langflow.api.v2 import files_router as files_router_v2
from langflow.api.v2 import mcp_router as mcp_router_v2
<<<<<<< Updated upstream
=======
from langflow.api.v1.events import router as events_router
# Import airelius_router from the fixed __init__.py to avoid circular imports
from langflow.airelius import PFUService, Retriever, PFU_KERNEL

# Create a simple router for airelius endpoints to avoid circular imports
from fastapi import APIRouter as FastAPIRouter
airelius_router = FastAPIRouter(prefix="/airelius", tags=["Airelius"])

# Add basic endpoint to test if the system is working
@airelius_router.get("/health")
async def airelius_health():
    """Check if Airelius PFU system is working."""
    return {"status": "healthy", "message": "Airelius PFU system is operational"}
>>>>>>> Stashed changes

router_v1 = APIRouter(
    prefix="/v1",
)

router_v2 = APIRouter(
    prefix="/v2",
)

router_v1.include_router(chat_router)
router_v1.include_router(endpoints_router)
router_v1.include_router(validate_router)
router_v1.include_router(store_router)
router_v1.include_router(flows_router)
router_v1.include_router(users_router)
router_v1.include_router(api_key_router)
router_v1.include_router(login_router)
router_v1.include_router(variables_router)
router_v1.include_router(files_router)
router_v1.include_router(monitor_router)
router_v1.include_router(folders_router)
router_v1.include_router(projects_router)
router_v1.include_router(starter_projects_router)
router_v1.include_router(mcp_router)
router_v1.include_router(voice_mode_router)
router_v1.include_router(mcp_projects_router)
<<<<<<< Updated upstream
=======
router_v1.include_router(events_router)
router_v1.include_router(airelius_router)
>>>>>>> Stashed changes

router_v2.include_router(files_router_v2)
router_v2.include_router(mcp_router_v2)

router = APIRouter(
    prefix="/api",
)
router.include_router(router_v1)
router.include_router(router_v2)
