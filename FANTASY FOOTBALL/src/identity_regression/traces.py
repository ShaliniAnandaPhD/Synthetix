"""
Platinum traces loader and sampler.
"""
import json
import os
import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_platinum_traces() -> Dict:
    """Load platinum traces from config."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config/platinum_traces.json")
    try:
        with open(config_path) as f:
            traces = json.load(f)
            logger.info(f"Loaded platinum traces: {len(traces)} keys")
            return traces
    except FileNotFoundError:
        logger.error(f"platinum_traces.json not found at {config_path}!")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in platinum_traces.json: {e}")
        return {}


def sample_platinum_traces(archetype: str, traces: Dict, n: int = 3) -> str:
    """
    Sample n random platinum traces for the archetype.
    
    Args:
        archetype: Archetype name (e.g., "statistician", "homer")
        traces: Dict of all platinum traces
        n: Number of examples to sample
    
    Returns:
        String with sampled traces joined by "---"
    """
    arch_traces = traces.get(archetype, [])
    
    if not arch_traces:
        logger.warning(f"No platinum traces for archetype '{archetype}'")
        return f"(No archived examples for {archetype})"
    
    sampled = random.sample(arch_traces, min(n, len(arch_traces)))
    return "\n---\n".join(sampled)


def validate_platinum_archive(traces: Dict, required_cities: List[str], required_events: List[str]) -> List[str]:
    """
    Validate that platinum archive has required coverage.
    
    Returns:
        List of missing keys (empty if valid)
    """
    missing = []
    
    for city in required_cities:
        for event in required_events:
            key = f"{city}:{event}"
            if key not in traces:
                missing.append(f"{key} (missing)")
            elif len(traces[key]) < 3:
                missing.append(f"{key} (only {len(traces[key])} traces, need 3+)")
    
    return missing
