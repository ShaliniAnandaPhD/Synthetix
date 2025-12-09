"""
Manual Override Controls

Allows creators to mute, skip, or pause agents during live commentary.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class OverrideType(Enum):
    """Types of manual overrides"""
    MUTE_AGENT = "mute_agent"       # Silence specific agent
    MUTE_REGION = "mute_region"     # Silence entire region
    SKIP_CURRENT = "skip_current"   # Skip current audio
    PAUSE_ALL = "pause_all"         # Pause all commentary
    RESUME = "resume"               # Resume from pause


@dataclass
class OverrideState:
    """Current override state for a session"""
    session_id: str
    muted_agents: Set[str] = field(default_factory=set)
    muted_regions: Set[str] = field(default_factory=set)
    is_paused: bool = False
    pause_start: Optional[float] = None
    overrides_count: int = 0


class ManualOverrideController:
    """
    Handles creator manual overrides during live commentary.
    
    Features:
    - Mute specific agents or regions
    - Skip current audio
    - Pause/resume all commentary
    
    Usage:
        controller = ManualOverrideController()
        
        # Creator wants to mute homer agent
        controller.mute_agent("session123", "homer")
        
        # Check if commentary should play
        if controller.should_play("session123", "kansas_city", "homer"):
            # Play the audio
            pass
    """
    
    def __init__(self):
        self.states: Dict[str, OverrideState] = {}
        self._skip_callbacks: Dict[str, callable] = {}
    
    def get_or_create_state(self, session_id: str) -> OverrideState:
        """Get or create override state for a session"""
        if session_id not in self.states:
            self.states[session_id] = OverrideState(session_id=session_id)
        return self.states[session_id]
    
    def mute_agent(self, session_id: str, agent_type: str) -> bool:
        """Mute a specific agent type"""
        state = self.get_or_create_state(session_id)
        state.muted_agents.add(agent_type)
        state.overrides_count += 1
        logger.info(f"Session {session_id}: Muted agent {agent_type}")
        return True
    
    def unmute_agent(self, session_id: str, agent_type: str) -> bool:
        """Unmute a specific agent type"""
        state = self.get_or_create_state(session_id)
        state.muted_agents.discard(agent_type)
        logger.info(f"Session {session_id}: Unmuted agent {agent_type}")
        return True
    
    def mute_region(self, session_id: str, region: str) -> bool:
        """Mute an entire region"""
        state = self.get_or_create_state(session_id)
        state.muted_regions.add(region)
        state.overrides_count += 1
        logger.info(f"Session {session_id}: Muted region {region}")
        return True
    
    def unmute_region(self, session_id: str, region: str) -> bool:
        """Unmute a region"""
        state = self.get_or_create_state(session_id)
        state.muted_regions.discard(region)
        logger.info(f"Session {session_id}: Unmuted region {region}")
        return True
    
    def skip_current(self, session_id: str) -> bool:
        """Skip the currently playing audio"""
        state = self.get_or_create_state(session_id)
        state.overrides_count += 1
        
        if session_id in self._skip_callbacks:
            self._skip_callbacks[session_id]()
            logger.info(f"Session {session_id}: Skipped current audio")
            return True
        return False
    
    def pause(self, session_id: str) -> bool:
        """Pause all commentary"""
        state = self.get_or_create_state(session_id)
        if not state.is_paused:
            state.is_paused = True
            state.pause_start = time.time()
            state.overrides_count += 1
            logger.info(f"Session {session_id}: Paused")
        return True
    
    def resume(self, session_id: str) -> bool:
        """Resume commentary"""
        state = self.get_or_create_state(session_id)
        if state.is_paused:
            state.is_paused = False
            pause_duration = time.time() - (state.pause_start or time.time())
            state.pause_start = None
            logger.info(f"Session {session_id}: Resumed after {pause_duration:.1f}s")
        return True
    
    def register_skip_callback(self, session_id: str, callback: callable):
        """Register a callback for skip events"""
        self._skip_callbacks[session_id] = callback
    
    def should_play(
        self, 
        session_id: str, 
        region: str, 
        agent_type: str
    ) -> bool:
        """
        Check if commentary should play based on current overrides.
        
        Returns True if the audio should be played.
        """
        if session_id not in self.states:
            return True  # No overrides, play everything
        
        state = self.states[session_id]
        
        # Check if paused
        if state.is_paused:
            return False
        
        # Check if region is muted
        if region in state.muted_regions:
            return False
        
        # Check if agent is muted
        if agent_type in state.muted_agents:
            return False
        
        return True
    
    def get_state(self, session_id: str) -> Optional[dict]:
        """Get current state for a session"""
        if session_id not in self.states:
            return None
        
        state = self.states[session_id]
        return {
            "session_id": session_id,
            "muted_agents": list(state.muted_agents),
            "muted_regions": list(state.muted_regions),
            "is_paused": state.is_paused,
            "pause_duration": (
                time.time() - state.pause_start 
                if state.is_paused and state.pause_start 
                else 0
            ),
            "overrides_count": state.overrides_count
        }
    
    def clear_state(self, session_id: str):
        """Clear all overrides for a session"""
        if session_id in self.states:
            del self.states[session_id]
        if session_id in self._skip_callbacks:
            del self._skip_callbacks[session_id]
    
    def handle_command(self, session_id: str, command: dict) -> dict:
        """
        Handle an override command from the client.
        
        Command format:
        {
            "action": "mute_agent" | "unmute_agent" | "mute_region" | ...
            "target": "agent_type" or "region_name" (optional)
        }
        """
        action = command.get("action")
        target = command.get("target")
        
        handlers = {
            "mute_agent": lambda: self.mute_agent(session_id, target),
            "unmute_agent": lambda: self.unmute_agent(session_id, target),
            "mute_region": lambda: self.mute_region(session_id, target),
            "unmute_region": lambda: self.unmute_region(session_id, target),
            "skip": lambda: self.skip_current(session_id),
            "pause": lambda: self.pause(session_id),
            "resume": lambda: self.resume(session_id),
        }
        
        if action in handlers:
            success = handlers[action]()
            return {
                "success": success,
                "action": action,
                "state": self.get_state(session_id)
            }
        
        return {"success": False, "error": f"Unknown action: {action}"}


# Singleton
_override_controller: Optional[ManualOverrideController] = None

def get_override_controller() -> ManualOverrideController:
    global _override_controller
    if _override_controller is None:
        _override_controller = ManualOverrideController()
    return _override_controller
