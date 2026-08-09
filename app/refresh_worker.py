"""Background worker to automatically refresh evergreen research runs."""
import asyncio
import logging
from app import db
from app.config import Settings, load_settings
from app.models import RunParams

log = logging.getLogger(__name__)

async def refresh_loop(orchestrator):
    """Periodically wakes up, finds evergreen runs, and spawns update runs."""
    cfg = load_settings()
    repo = db.Repo(db.connect(cfg.db_path))
    
    while True:
        try:
            # Wake up every 24 hours
            await asyncio.sleep(24 * 3600)
            
            evergreen_runs = repo.list_evergreen_runs()
            if not evergreen_runs:
                continue
                
            log.info("Waking up to refresh %d evergreen runs", len(evergreen_runs))
            
            for row in evergreen_runs:
                # Disable evergreen on the parent so we don't branch indefinitely from it
                repo.update_run(row["id"], evergreen=False)
                
                # Spawn a new run targeting the past month, building on the parent
                params = RunParams(
                    query=row["query"],
                    depth=row["depth"],
                    recency="month",
                    parent_run_id=row["id"],
                    evergreen=True,
                    origin="web",
                    origin_chat_id=row["origin_chat_id"]
                )
                new_run_id = orchestrator.enqueue(params)
                log.info("Spawned evergreen refresh run %s for parent %s", new_run_id, row["id"])
                
        except asyncio.CancelledError:
            log.info("Refresh loop shutting down")
            break
        except Exception as e:
            log.exception("Error in refresh loop: %s", e)
            await asyncio.sleep(300)  # wait 5 mins before retry on crash
