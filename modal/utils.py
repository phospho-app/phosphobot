from supabase import Client
from loguru import logger
from datetime import datetime, timezone


def update_server_status(
    supabase_client: Client,
    server_id: int,
    status: str,
):
    logger.info(f"Updating server status to {status} for server_id {server_id}")
    if status == "failed":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    elif status == "stopped":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    else:
        raise NotImplementedError(
            f"Status '{status}' not implemented for server update"
        )
