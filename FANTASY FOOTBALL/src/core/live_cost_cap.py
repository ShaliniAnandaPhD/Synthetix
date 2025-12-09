"""
Live Cost Cap Manager

Tracks and limits spend during live game commentary.
Prevents runaway costs from loops or high volume.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs"""
    LLM = "llm"              # Vertex AI / Gemini
    TTS = "tts"              # ElevenLabs / Google TTS
    CACHE = "cache"          # Redis
    COMPUTE = "compute"      # Modal compute


@dataclass
class CostConfig:
    """Cost configuration per game"""
    # Hard limits
    max_per_game_usd: float = 25.0
    max_per_hour_usd: float = 10.0
    max_per_minute_usd: float = 0.5
    
    # Warning thresholds (as percentage of limit)
    warn_threshold: float = 0.75  # 75%
    alert_threshold: float = 0.90  # 90%
    
    # Per-unit costs (estimates)
    llm_cost_per_1k_tokens: float = 0.00015
    tts_cost_per_char_elevenlabs: float = 0.0003
    tts_cost_per_char_google: float = 0.000016
    modal_cost_per_second: float = 0.0001


@dataclass
class CostSample:
    """A cost incurred"""
    category: CostCategory
    amount_usd: float
    timestamp: float = field(default_factory=time.time)
    description: str = ""


class LiveCostCap:
    """
    Tracks and enforces cost limits during live commentary.
    
    Features:
    - Per-game, per-hour, per-minute limits
    - Warning and alert thresholds
    - Automatic throttling when near limits
    - Cost reporting by category
    
    Usage:
        cap = LiveCostCap(game_id="KC_BUF")
        
        # Before each operation
        if cap.can_spend(CostCategory.LLM, 0.01):
            # Do the operation
            cap.record_cost(CostCategory.LLM, 0.01, "Agent generation")
        else:
            # Skip or use fallback
            pass
        
        # Get status
        status = cap.get_status()
    """
    
    def __init__(
        self,
        game_id: str,
        config: Optional[CostConfig] = None,
        on_warn: Optional[Callable[[str], None]] = None,
        on_limit: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize cost cap.
        
        Args:
            game_id: Identifier for the game
            config: Cost configuration
            on_warn: Callback when approaching limit
            on_limit: Callback when limit reached
        """
        self.game_id = game_id
        self.config = config or CostConfig()
        self.on_warn = on_warn
        self.on_limit = on_limit
        
        self._costs: list[CostSample] = []
        self._start_time = time.time()
        self._warned = False
        self._alerted = False
        self._limited = False
    
    @property
    def total_cost(self) -> float:
        """Total cost so far"""
        return sum(c.amount_usd for c in self._costs)
    
    @property
    def cost_last_hour(self) -> float:
        """Cost in the last hour"""
        cutoff = time.time() - 3600
        return sum(c.amount_usd for c in self._costs if c.timestamp > cutoff)
    
    @property
    def cost_last_minute(self) -> float:
        """Cost in the last minute"""
        cutoff = time.time() - 60
        return sum(c.amount_usd for c in self._costs if c.timestamp > cutoff)
    
    def can_spend(self, category: CostCategory, amount_usd: float) -> bool:
        """
        Check if we can afford to spend this amount.
        Returns False if it would exceed any limit.
        """
        # Check per-game limit
        if self.total_cost + amount_usd > self.config.max_per_game_usd:
            self._trigger_limit("game limit")
            return False
        
        # Check per-hour limit
        if self.cost_last_hour + amount_usd > self.config.max_per_hour_usd:
            self._trigger_limit("hourly limit")
            return False
        
        # Check per-minute limit
        if self.cost_last_minute + amount_usd > self.config.max_per_minute_usd:
            self._trigger_limit("minute limit")
            return False
        
        return True
    
    def record_cost(
        self, 
        category: CostCategory, 
        amount_usd: float,
        description: str = ""
    ):
        """Record a cost that was incurred"""
        sample = CostSample(
            category=category,
            amount_usd=amount_usd,
            description=description
        )
        self._costs.append(sample)
        
        # Check warning thresholds
        self._check_thresholds()
    
    def estimate_llm_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate LLM cost"""
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * self.config.llm_cost_per_1k_tokens
    
    def estimate_tts_cost(self, text_length: int, use_elevenlabs: bool = True) -> float:
        """Estimate TTS cost"""
        if use_elevenlabs:
            return text_length * self.config.tts_cost_per_char_elevenlabs
        return text_length * self.config.tts_cost_per_char_google
    
    def get_cost_by_category(self) -> Dict[str, float]:
        """Get costs broken down by category"""
        breakdown: Dict[str, float] = {}
        for sample in self._costs:
            key = sample.category.value
            breakdown[key] = breakdown.get(key, 0) + sample.amount_usd
        return breakdown
    
    def get_status(self) -> dict:
        """Get current cost status"""
        return {
            "game_id": self.game_id,
            "total_cost_usd": round(self.total_cost, 4),
            "cost_last_hour_usd": round(self.cost_last_hour, 4),
            "cost_last_minute_usd": round(self.cost_last_minute, 6),
            "percent_of_game_limit": round(
                self.total_cost / self.config.max_per_game_usd * 100, 1
            ),
            "percent_of_hour_limit": round(
                self.cost_last_hour / self.config.max_per_hour_usd * 100, 1
            ),
            "by_category": self.get_cost_by_category(),
            "is_limited": self._limited,
            "duration_seconds": int(time.time() - self._start_time),
        }
    
    def get_remaining_budget(self) -> dict:
        """Get remaining budget for each time window"""
        return {
            "game_remaining": max(0, self.config.max_per_game_usd - self.total_cost),
            "hour_remaining": max(0, self.config.max_per_hour_usd - self.cost_last_hour),
            "minute_remaining": max(0, self.config.max_per_minute_usd - self.cost_last_minute),
        }
    
    def reset(self):
        """Reset all costs (for new game)"""
        self._costs.clear()
        self._start_time = time.time()
        self._warned = False
        self._alerted = False
        self._limited = False
    
    def _check_thresholds(self):
        """Check warning and alert thresholds"""
        percent = self.total_cost / self.config.max_per_game_usd
        
        if not self._warned and percent >= self.config.warn_threshold:
            self._warned = True
            msg = f"Approaching cost limit: {percent:.0%} of ${self.config.max_per_game_usd}"
            logger.warning(msg)
            if self.on_warn:
                self.on_warn(msg)
        
        if not self._alerted and percent >= self.config.alert_threshold:
            self._alerted = True
            msg = f"Near cost limit: {percent:.0%} of ${self.config.max_per_game_usd}"
            logger.error(msg)
            if self.on_warn:
                self.on_warn(msg)
    
    def _trigger_limit(self, reason: str):
        """Trigger limit reached"""
        if not self._limited:
            self._limited = True
            msg = f"Cost limit reached: {reason} (total: ${self.total_cost:.4f})"
            logger.error(msg)
            if self.on_limit:
                self.on_limit(msg)


# Factory function with reasonable defaults
def create_cost_cap(
    game_id: str,
    max_per_game: float = 25.0,
    max_per_hour: float = 10.0,
    max_per_minute: float = 5.0,  # Higher default for testing
    on_warn: Optional[Callable] = None,
    on_limit: Optional[Callable] = None,
) -> LiveCostCap:
    """Create a configured cost cap"""
    config = CostConfig(
        max_per_game_usd=max_per_game,
        max_per_hour_usd=max_per_hour,
        max_per_minute_usd=max_per_minute,
    )
    return LiveCostCap(
        game_id=game_id,
        config=config,
        on_warn=on_warn,
        on_limit=on_limit,
    )


if __name__ == "__main__":
    warnings = []
    limits = []
    
    cap = LiveCostCap(
        game_id="KC_BUF",
        config=CostConfig(max_per_game_usd=1.0),  # Low for testing
        on_warn=lambda msg: warnings.append(msg),
        on_limit=lambda msg: limits.append(msg),
    )
    
    # Simulate costs
    for i in range(20):
        amount = 0.05
        if cap.can_spend(CostCategory.LLM, amount):
            cap.record_cost(CostCategory.LLM, amount, f"Request {i}")
            print(f"Request {i}: ${cap.total_cost:.4f}")
        else:
            print(f"Request {i}: BLOCKED")
            break
    
    print(f"\nStatus: {cap.get_status()}")
    print(f"Warnings: {warnings}")
    print(f"Limits: {limits}")
