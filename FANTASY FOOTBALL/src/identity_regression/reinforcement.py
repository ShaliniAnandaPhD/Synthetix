"""
Identity reinforcement for failed vibe checks.
"""
import json
from typing import Dict
from .traces import sample_platinum_traces
from .vibe_check import load_archetype_config


def reinforce_identity(prompt: str, archetype: str, platinum_traces: Dict) -> str:
    """
    Reinforce identity when vibe check fails.
    
    Injects platinum examples and explicit style instructions
    to encourage the model to stay in character.
    
    Args:
        prompt: Original prompt
        archetype: Archetype that needs reinforcement
        platinum_traces: Dict of platinum traces
    
    Returns:
        Reinforced prompt with examples and style instructions
    """
    platinum_examples = sample_platinum_traces(archetype, platinum_traces, n=3)
    config = load_archetype_config().get(archetype, {})
    
    archetype_name = config.get("name", archetype.title())
    examples = config.get("examples", [])
    phrases = config.get("signature_phrases", [])
    energy = config.get("energy_baseline", 0.5)
    
    reinforced = f"""⚠️ IDENTITY ALERT: Your previous response was too neutral/generic.

You are {archetype_name}. 
Your style reference: {', '.join(examples)}

REQUIRED SIGNATURE PHRASES (use at least 2):
{json.dumps(phrases, indent=2)}

ENERGY LEVEL: {energy:.0%} intensity
(0.4 = calm/analytical, 0.95 = explosive/emphatic)

HERE ARE EXAMPLES OF AUTHENTIC {archetype.upper()} RESPONSES:
---
{platinum_examples}
---

NOW RESPOND IN THIS EXACT STYLE. BE BOLD. STAY IN CHARACTER:
{prompt}"""
    
    return reinforced
