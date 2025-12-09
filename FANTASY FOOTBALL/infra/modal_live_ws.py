"""
Live Commentary WebSocket Server

Modal-hosted WebSocket endpoint for real-time game commentary streaming.
Connects to NFL event feed and dispatches to agents for live reactions.
"""

import modal
import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Set, Optional
from dataclasses import dataclass, field

app = modal.App("neuron-live-ws")

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "websockets",
        "redis",
        "aiohttp",
        "google-cloud-aiplatform",
    )
    .add_local_dir(
        local_path="src",
        remote_path="/root/src"
    )
    .add_local_dir(
        local_path="config",
        remote_path="/root/config"
    )
)


# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("redis-credentials"),
        modal.Secret.from_name("googlecloud-secret"),
    ],
    timeout=14400,  # 4 hours max per session
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def live_server():
    """
    Modal ASGI endpoint for WebSocket server.
    
    Deploy: modal deploy infra/modal_live_ws.py
    URL: https://neuron-live-ws-*.modal.run
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
    from fastapi.middleware.cors import CORSMiddleware
    
    fastapi_app = FastAPI(title="Neuron Live Commentary")
    
    # Add CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Session storage
    sessions: Dict[str, dict] = {}
    game_sessions: Dict[str, Set[str]] = {}
    
    @fastapi_app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "active_sessions": len(sessions),
            "timestamp": datetime.now().isoformat()
        }
    
    @fastapi_app.get("/sessions")
    async def list_sessions():
        return {
            "sessions": [
                {
                    "session_id": s["session_id"],
                    "game_id": s["game_id"],
                    "regions": s["regions"],
                    "events_received": s.get("events_received", 0),
                    "commentary_sent": s.get("commentary_sent", 0),
                    "uptime_seconds": int(time.time() - s.get("started_at", time.time()))
                }
                for s in sessions.values()
            ]
        }
    
    async def mock_event_stream(game_id: str):
        """Generate mock NFL events for testing"""
        mock_events = [
            {"type": "play", "description": "First down run for 5 yards", "team": "KC"},
            {"type": "big_play", "description": "Mahomes to Kelce for 35 yards!", "team": "KC"},
            {"type": "touchdown", "description": "TOUCHDOWN! Mahomes to Kelce!", "team": "KC"},
            {"type": "play", "description": "Kickoff returned to the 25", "team": "BUF"},
            {"type": "turnover", "description": "Josh Allen INTERCEPTED!", "team": "BUF"},
        ]
        
        for event in mock_events:
            await asyncio.sleep(5)
            event["game_id"] = game_id
            event["timestamp"] = time.time()
            yield event
    
    async def generate_mock_commentary(event: dict, regions: list) -> list:
        """Generate mock commentary"""
        event_type = event.get("type", "play")
        
        reactions = {
            "touchdown": [
                ("homer", "TOUCHDOWN! That's what I'm talking about! 🏈"),
                ("analyst", "Perfect execution on that drive. Six points."),
            ],
            "turnover": [
                ("homer", "NO! You've got to be kidding me!"),
                ("analyst", "That's a killer. Momentum shift coming."),
            ],
            "big_play": [
                ("homer", "WHAT A PLAY! Did you see that?!"),
                ("analyst", "Great scheme recognition there."),
            ],
        }
        
        phrases = reactions.get(event_type, [("analyst", f"Play continues...")])
        
        commentaries = []
        for region in regions[:2]:
            for agent_type, text in phrases:
                commentaries.append({
                    "type": "commentary",
                    "region": region,
                    "agent": agent_type,
                    "text": text,
                    "emotion": "excited" if event_type == "touchdown" else "analytical",
                    "timestamp": time.time()
                })
        
        return commentaries
    
    @fastapi_app.websocket("/live/{game_id}")
    async def live_commentary_stream(
        websocket: WebSocket, 
        game_id: str,
        creator_id: str = Query(default="anonymous"),
        regions: str = Query(default="kansas_city,dallas")
    ):
        await websocket.accept()
        
        region_list = [r.strip() for r in regions.split(",")]
        session_id = f"{creator_id}_{game_id}_{int(time.time())}"
        
        session = {
            "session_id": session_id,
            "creator_id": creator_id,
            "game_id": game_id,
            "regions": region_list,
            "started_at": time.time(),
            "events_received": 0,
            "commentary_sent": 0
        }
        sessions[session_id] = session
        
        if game_id not in game_sessions:
            game_sessions[game_id] = set()
        game_sessions[game_id].add(session_id)
        
        await websocket.send_json({
            "type": "session_start",
            "session_id": session_id,
            "game_id": game_id,
            "regions": region_list,
            "timestamp": time.time()
        })
        
        try:
            async for event in mock_event_stream(game_id):
                session["events_received"] += 1
                
                await websocket.send_json({
                    "type": "event",
                    "data": event
                })
                
                commentaries = await generate_mock_commentary(event, region_list)
                
                for commentary in commentaries:
                    await websocket.send_json(commentary)
                    session["commentary_sent"] += 1
                    await asyncio.sleep(0.5)
        
        except WebSocketDisconnect:
            print(f"Session {session_id} disconnected")
        
        except Exception as e:
            print(f"Session {session_id} error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass
        
        finally:
            duration = int(time.time() - session["started_at"])
            print(f"Session {session_id} ended after {duration}s")
            
            sessions.pop(session_id, None)
            if game_id in game_sessions:
                game_sessions[game_id].discard(session_id)
            
            try:
                await websocket.send_json({
                    "type": "session_end",
                    "session_id": session_id,
                    "stats": {
                        "duration_seconds": duration,
                        "events_received": session["events_received"],
                        "commentary_sent": session["commentary_sent"]
                    }
                })
            except:
                pass
    
    return fastapi_app


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("NEURON LIVE COMMENTARY - WEBSOCKET SERVER")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /sessions        - List active sessions")
    print("  WS   /live/{game_id}  - Live commentary stream")
    print("\nQuery params for /live/{game_id}:")
    print("  creator_id - Your creator ID")
    print("  regions    - Comma-separated regions")
    print("\n" + "=" * 60)
