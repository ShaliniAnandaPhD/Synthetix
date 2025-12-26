"""
Event type mapping from ESPN to fallback categories.
"""
from typing import Dict


def map_event_to_type(event: Dict) -> str:
    """
    Map ESPN event data to fallback event type.
    
    Args:
        event: ESPN event data dict with "type" key
    
    Returns:
        Event type string for platinum fallback lookup
    """
    # Handle various ESPN event formats
    play_type = ""
    
    if isinstance(event.get("type"), dict):
        play_type = event.get("type", {}).get("text", "").lower()
    elif isinstance(event.get("type"), str):
        play_type = event.get("type", "").lower()
    elif event.get("text"):
        play_type = event.get("text", "").lower()
    elif event.get("description"):
        play_type = event.get("description", "").lower()
    
    # Event mapping - check for keywords
    event_mapping = {
        # Touchdowns
        "touchdown": "touchdown",
        "td": "touchdown",
        "passing touchdown": "touchdown",
        "rushing touchdown": "touchdown",
        "receiving touchdown": "touchdown",
        "punt return touchdown": "touchdown",
        "kick return touchdown": "touchdown",
        "defensive touchdown": "touchdown",
        "pick six": "touchdown",
        "fumble return touchdown": "touchdown",
        
        # Turnovers
        "interception": "interception",
        "int": "interception",
        "intercepted": "interception",
        "fumble": "fumble",
        "fumble recovery": "fumble",
        "lost fumble": "fumble",
        "turnover": "interception",
        
        # Field goals
        "field goal": "field_goal",
        "fg": "field_goal",
        "field goal good": "field_goal",
        "field goal missed": "field_goal",
        
        # Defense
        "sack": "sack",
        "sacked": "sack",
        
        # Penalties
        "penalty": "penalty",
        "flag": "penalty",
        "holding": "penalty",
        "pass interference": "penalty",
        "offsides": "penalty",
        "false start": "penalty",
        
        # Big plays
        "big play": "big_play",
        "long gain": "big_play",
        "deep pass": "big_play",
        
        # Stoppages
        "timeout": "timeout",
        "challenge": "challenge",
        "injury": "injury",
        "injury timeout": "injury",
        
        # Misc
        "end of quarter": "generic",
        "end of half": "generic",
        "two minute warning": "generic",
    }
    
    # Check each mapping
    for key, event_type in event_mapping.items():
        if key in play_type:
            return event_type
    
    return "generic"


def get_event_importance(event_type: str) -> int:
    """
    Get importance score for event type (1-10).
    Used to prioritize which events get commentary.
    """
    importance = {
        "touchdown": 10,
        "interception": 9,
        "fumble": 9,
        "field_goal": 6,
        "sack": 5,
        "big_play": 7,
        "penalty": 4,
        "timeout": 2,
        "challenge": 3,
        "injury": 3,
        "generic": 1
    }
    return importance.get(event_type, 1)
