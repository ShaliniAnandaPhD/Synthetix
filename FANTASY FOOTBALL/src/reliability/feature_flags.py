"""
Feature Flags

Toggle features without redeployment.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class FlagType(Enum):
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"


@dataclass
class FeatureFlag:
    """A single feature flag"""
    name: str
    description: str = ""
    flag_type: FlagType = FlagType.BOOLEAN
    enabled: bool = False
    percentage: float = 0  # For percentage rollout
    user_list: List[str] = field(default_factory=list)  # For user targeting
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class FeatureFlagManager:
    """
    Feature flag management for toggling features.
    
    Usage:
        flags = FeatureFlagManager()
        
        # Define flags
        flags.define("new_live_commentary", "New live commentary engine", enabled=False)
        flags.define("elevenlabs_v2", "Use ElevenLabs V2 API", percentage=10)
        
        # Check flags
        if flags.is_enabled("new_live_commentary"):
            use_new_engine()
        
        # Check with user context
        if flags.is_enabled_for("elevenlabs_v2", user_id="creator123"):
            use_v2_api()
    """
    
    def __init__(self, storage_client=None):
        self.storage = storage_client
        self._flags: Dict[str, FeatureFlag] = {}
        self._overrides: Dict[str, Dict[str, bool]] = {}  # flag_name -> {user_id: enabled}
    
    def define(
        self,
        name: str,
        description: str = "",
        flag_type: FlagType = FlagType.BOOLEAN,
        enabled: bool = False,
        percentage: float = 0,
        user_list: List[str] = None
    ) -> FeatureFlag:
        """Define a new feature flag"""
        flag = FeatureFlag(
            name=name,
            description=description,
            flag_type=flag_type,
            enabled=enabled,
            percentage=percentage,
            user_list=user_list or []
        )
        self._flags[name] = flag
        return flag
    
    def enable(self, name: str):
        """Enable a flag globally"""
        if name in self._flags:
            self._flags[name].enabled = True
            self._flags[name].updated_at = time.time()
            logger.info(f"Flag enabled: {name}")
    
    def disable(self, name: str):
        """Disable a flag globally"""
        if name in self._flags:
            self._flags[name].enabled = False
            self._flags[name].updated_at = time.time()
            logger.info(f"Flag disabled: {name}")
    
    def set_percentage(self, name: str, percentage: float):
        """Set percentage rollout for a flag"""
        if name in self._flags:
            self._flags[name].percentage = max(0, min(100, percentage))
            self._flags[name].flag_type = FlagType.PERCENTAGE
            self._flags[name].updated_at = time.time()
            logger.info(f"Flag {name} set to {percentage}% rollout")
    
    def add_user(self, name: str, user_id: str):
        """Add user to flag allowlist"""
        if name in self._flags:
            if user_id not in self._flags[name].user_list:
                self._flags[name].user_list.append(user_id)
    
    def remove_user(self, name: str, user_id: str):
        """Remove user from flag allowlist"""
        if name in self._flags:
            if user_id in self._flags[name].user_list:
                self._flags[name].user_list.remove(user_id)
    
    def override(self, name: str, user_id: str, enabled: bool):
        """Set user-specific override"""
        if name not in self._overrides:
            self._overrides[name] = {}
        self._overrides[name][user_id] = enabled
    
    def is_enabled(self, name: str) -> bool:
        """Check if flag is enabled globally"""
        if name not in self._flags:
            return False
        return self._flags[name].enabled
    
    def is_enabled_for(self, name: str, user_id: str = "") -> bool:
        """Check if flag is enabled for a specific user"""
        if name not in self._flags:
            return False
        
        flag = self._flags[name]
        
        # Check user override first
        if name in self._overrides and user_id in self._overrides[name]:
            return self._overrides[name][user_id]
        
        # Check by flag type
        if flag.flag_type == FlagType.BOOLEAN:
            return flag.enabled
        
        elif flag.flag_type == FlagType.USER_LIST:
            return user_id in flag.user_list
        
        elif flag.flag_type == FlagType.PERCENTAGE:
            # Consistent hashing for percentage rollout
            import hashlib
            hash_input = f"{name}:{user_id}".encode()
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
            bucket = hash_value % 100
            return bucket < flag.percentage
        
        return False
    
    def get_flag(self, name: str) -> Optional[dict]:
        """Get flag details"""
        if name not in self._flags:
            return None
        
        f = self._flags[name]
        return {
            "name": f.name,
            "description": f.description,
            "type": f.flag_type.value,
            "enabled": f.enabled,
            "percentage": f.percentage,
            "user_count": len(f.user_list),
            "updated_at": f.updated_at
        }
    
    def get_all_flags(self) -> List[dict]:
        """Get all flags"""
        return [self.get_flag(name) for name in self._flags]
    
    def get_enabled_flags(self) -> List[str]:
        """Get list of enabled flag names"""
        return [name for name, flag in self._flags.items() if flag.enabled]


# Default flags for Fantasy Football Neuron
DEFAULT_FLAGS = {
    "live_commentary_v2": ("New live commentary engine", False),
    "elevenlabs_turbo": ("Use ElevenLabs Turbo voices", False),
    "vertex_caching": ("Enable Vertex AI context caching", True),
    "phrase_cache": ("Pre-generated phrase caching", True),
    "cost_alerts": ("Enable cost alerts", True),
    "multi_creator_sync": ("Multi-creator game sync", False),
    "mobile_ui": ("Mobile-optimized UI", False),
}


# Singleton
_flag_manager: Optional[FeatureFlagManager] = None

def get_feature_flags() -> FeatureFlagManager:
    global _flag_manager
    if _flag_manager is None:
        _flag_manager = FeatureFlagManager()
        # Register default flags
        for name, (desc, enabled) in DEFAULT_FLAGS.items():
            _flag_manager.define(name, desc, enabled=enabled)
    return _flag_manager
