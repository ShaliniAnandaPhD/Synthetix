"""
Per-Debate Cost Attribution

Tracks exact cost per debate with LLM and TTS breakdown.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CostType(Enum):
    """Types of costs"""
    LLM_INPUT = "llm_input"
    LLM_OUTPUT = "llm_output"
    TTS_ELEVENLABS = "tts_elevenlabs"
    TTS_GOOGLE = "tts_google"
    STORAGE = "storage"
    COMPUTE = "compute"


@dataclass
class CostEntry:
    """A single cost entry"""
    cost_type: CostType
    amount_usd: float
    units: int  # tokens or characters
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class DebateCost:
    """Cost breakdown for a single debate"""
    debate_id: str
    creator_id: str
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    entries: List[CostEntry] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        return sum(e.amount_usd for e in self.entries)
    
    @property
    def llm_cost(self) -> float:
        return sum(
            e.amount_usd for e in self.entries 
            if e.cost_type in (CostType.LLM_INPUT, CostType.LLM_OUTPUT)
        )
    
    @property
    def tts_cost(self) -> float:
        return sum(
            e.amount_usd for e in self.entries 
            if e.cost_type in (CostType.TTS_ELEVENLABS, CostType.TTS_GOOGLE)
        )
    
    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "creator_id": self.creator_id,
            "total_cost_usd": round(self.total_cost, 6),
            "llm_cost_usd": round(self.llm_cost, 6),
            "tts_cost_usd": round(self.tts_cost, 6),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": (
                int((self.completed_at or time.time()) - self.started_at)
            ),
            "entries": [
                {
                    "type": e.cost_type.value,
                    "amount_usd": round(e.amount_usd, 6),
                    "units": e.units,
                    "description": e.description
                }
                for e in self.entries
            ]
        }


class DebateCostTracker:
    """
    Tracks exact cost per debate.
    
    Pricing (as of Dec 2024):
    - Gemini 1.5 Flash: $0.075/1M input, $0.30/1M output
    - ElevenLabs: ~$0.30/1000 chars
    - Google TTS: ~$0.016/1000 chars
    
    Usage:
        tracker = DebateCostTracker()
        
        # Start tracking for debate
        tracker.start_debate("debate123", "creator456")
        
        # Record costs during generation
        tracker.record_llm_usage("debate123", input_tokens=500, output_tokens=200)
        tracker.record_tts_usage("debate123", chars=150, provider="elevenlabs")
        
        # Complete and get summary
        costs = tracker.complete_debate("debate123")
    """
    
    # Pricing per unit
    PRICING = {
        "gemini_flash_input_per_1m": 0.075,
        "gemini_flash_output_per_1m": 0.30,
        "elevenlabs_per_1k_chars": 0.30,
        "google_tts_per_1k_chars": 0.016,
        "modal_per_second": 0.0001,
    }
    
    def __init__(self):
        self._active_debates: Dict[str, DebateCost] = {}
        self._completed_debates: Dict[str, DebateCost] = {}
    
    def start_debate(self, debate_id: str, creator_id: str) -> DebateCost:
        """Start tracking costs for a new debate"""
        debate = DebateCost(
            debate_id=debate_id,
            creator_id=creator_id
        )
        self._active_debates[debate_id] = debate
        return debate
    
    def record_llm_usage(
        self, 
        debate_id: str, 
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini_flash"
    ):
        """Record LLM token usage"""
        if debate_id not in self._active_debates:
            return
        
        debate = self._active_debates[debate_id]
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * self.PRICING["gemini_flash_input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * self.PRICING["gemini_flash_output_per_1m"]
        
        debate.entries.append(CostEntry(
            cost_type=CostType.LLM_INPUT,
            amount_usd=input_cost,
            units=input_tokens,
            description=f"{model} input"
        ))
        
        debate.entries.append(CostEntry(
            cost_type=CostType.LLM_OUTPUT,
            amount_usd=output_cost,
            units=output_tokens,
            description=f"{model} output"
        ))
    
    def record_tts_usage(
        self, 
        debate_id: str, 
        chars: int,
        provider: str = "elevenlabs"
    ):
        """Record TTS character usage"""
        if debate_id not in self._active_debates:
            return
        
        debate = self._active_debates[debate_id]
        
        if provider == "elevenlabs":
            cost = (chars / 1000) * self.PRICING["elevenlabs_per_1k_chars"]
            cost_type = CostType.TTS_ELEVENLABS
        else:
            cost = (chars / 1000) * self.PRICING["google_tts_per_1k_chars"]
            cost_type = CostType.TTS_GOOGLE
        
        debate.entries.append(CostEntry(
            cost_type=cost_type,
            amount_usd=cost,
            units=chars,
            description=f"{provider} TTS"
        ))
    
    def record_compute_usage(self, debate_id: str, seconds: float):
        """Record Modal compute time"""
        if debate_id not in self._active_debates:
            return
        
        debate = self._active_debates[debate_id]
        cost = seconds * self.PRICING["modal_per_second"]
        
        debate.entries.append(CostEntry(
            cost_type=CostType.COMPUTE,
            amount_usd=cost,
            units=int(seconds),
            description="Modal compute"
        ))
    
    def complete_debate(self, debate_id: str) -> Optional[dict]:
        """Complete tracking and return cost summary"""
        if debate_id not in self._active_debates:
            return None
        
        debate = self._active_debates.pop(debate_id)
        debate.completed_at = time.time()
        self._completed_debates[debate_id] = debate
        
        return debate.to_dict()
    
    def get_debate_cost(self, debate_id: str) -> Optional[dict]:
        """Get cost breakdown for a debate"""
        debate = (
            self._active_debates.get(debate_id) or 
            self._completed_debates.get(debate_id)
        )
        if debate:
            return debate.to_dict()
        return None
    
    def get_creator_total(self, creator_id: str) -> dict:
        """Get total costs for a creator"""
        all_debates = list(self._active_debates.values()) + list(self._completed_debates.values())
        creator_debates = [d for d in all_debates if d.creator_id == creator_id]
        
        total = sum(d.total_cost for d in creator_debates)
        llm = sum(d.llm_cost for d in creator_debates)
        tts = sum(d.tts_cost for d in creator_debates)
        
        return {
            "creator_id": creator_id,
            "debate_count": len(creator_debates),
            "total_cost_usd": round(total, 4),
            "llm_cost_usd": round(llm, 4),
            "tts_cost_usd": round(tts, 4),
            "avg_per_debate": round(total / len(creator_debates), 4) if creator_debates else 0
        }
    
    def get_summary(self) -> dict:
        """Get overall cost summary"""
        all_debates = list(self._completed_debates.values())
        
        if not all_debates:
            return {"total_debates": 0, "total_cost_usd": 0}
        
        total = sum(d.total_cost for d in all_debates)
        llm = sum(d.llm_cost for d in all_debates)
        tts = sum(d.tts_cost for d in all_debates)
        
        return {
            "total_debates": len(all_debates),
            "active_debates": len(self._active_debates),
            "total_cost_usd": round(total, 2),
            "llm_cost_usd": round(llm, 2),
            "tts_cost_usd": round(tts, 2),
            "avg_per_debate": round(total / len(all_debates), 4),
            "llm_percent": round(llm / total * 100, 1) if total > 0 else 0,
            "tts_percent": round(tts / total * 100, 1) if total > 0 else 0,
        }


# Singleton
_tracker: Optional[DebateCostTracker] = None

def get_cost_tracker() -> DebateCostTracker:
    global _tracker
    if _tracker is None:
        _tracker = DebateCostTracker()
    return _tracker
