from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from langflow.api.utils import DbSession
from langflow.services.auth.utils import get_current_user_for_websocket


router = APIRouter(prefix="/events", tags=["Events"])


# Keep track of connected clients. This is per-process.
connected_clients: Set[WebSocket] = set()


@router.websocket("/ws")
async def events_ws(ws: WebSocket, session: DbSession):
    """Accepts a WebSocket connection for broadcasting server events.

    Authenticates using the same mechanism as other WebSocket endpoints.
    """
    try:
        await get_current_user_for_websocket(ws, session)
    except Exception:
        await ws.close(code=1008)
        return

    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            # Keep the connection alive. We do not expect messages from the client.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)


async def broadcast_flow_updated(flow_id: str) -> None:
    """Broadcast a flow-updated event to all connected clients.

    Parameters
    ----------
    flow_id: str
        The ID of the flow that has been updated.
    """
    dead_clients: list[WebSocket] = []
    for ws in list(connected_clients):
        try:
            await ws.send_json({"type": "flow_updated", "flow_id": flow_id})
        except Exception:
            dead_clients.append(ws)

    # Cleanup any broken connections
    for ws in dead_clients:
        connected_clients.discard(ws)


