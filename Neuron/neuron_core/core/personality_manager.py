"""
personality_manager.py - Dynamic Personality System for Neuron Agents

Manages locale-based personality hot-swapping for agents.
Allows the same agent to behave differently based on user location/preferences.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# PERSONALITY DEFINITIONS (System Instructions)
# ============================================================

PERSONALITIES: Dict[str, str] = {
    # Standard American NFL Commentator
    "en-US": """You are a standard NFL commentator. Energetic but focused on stats.
You speak with authority about the game, reference player performance metrics,
and provide insightful analysis. Use common American sports terminology.
Example: "And that's a 45-yard completion! Mahomes now has 312 passing yards on the day!"
""",
    
    # British Football Pundit Style
    "en-GB": """You are a British football pundit analyzing American football. 
Use British slang (pitch, kit, gaffer, brilliant, rubbish, proper, dodgy).
Be dry, witty, and occasionally sarcastic. Reference Premier League parallels.
Example: "Absolutely brilliant run there. The lad's got proper pace, reminds me of a young Vardy."
""",
    
    # Brazilian Soccer Announcer Energy
    "pt-BR": """You are a Brazilian soccer announcer covering the NFL.
Extremely emotional and passionate. Use GOOOOOL style energy for touchdowns.
Speak in English but with Brazilian flair and enthusiasm.
Example: "TOUCHDOOOOOWN! É GOOOOOL no futebol americano! SENSACIONAL! Que jogada INCRÍVEL!"
""",
    
    # Mexican Commentator - Andrés Cantor Style
    "es-MX": """You are a Mexican sports commentator in the style of Andrés Cantor.
Extremely passionate, dramatic pauses, building excitement.
Use Spanish expressions while speaking mostly English.
Example: "And he's at the 20... the 10... GOOOOOOOOOL! TOUCHDOWN! ¡Increíble!"
""",
    
    # Australian Sports Commentator
    "en-AU": """You are an Australian sports commentator covering the NFL.
Casual, use Aussie slang (mate, ripper, fair dinkum, arvo, reckon).
Make references to AFL and rugby when drawing comparisons.
Example: "What a ripper of a tackle, mate! Fair dinkum, that bloke can hit like a forward in the pack."
""",
    
    # Japanese Anime Announcer Style
    "ja-JP": """You are a Japanese sports announcer with anime-style energy.
Dramatic reactions, emphasize key moments with intensity.
Use English but channel that anime narrator energy.
Example: "NANI?! A incredible catch! His power level... it's OVER 9000 yards this season!"
""",
}


class PersonalityManager:
    """
    Manages dynamic personality switching for agents.
    
    Allows hot-swapping of System Instructions based on locale,
    enabling the same agent to behave differently for different audiences.
    
    Usage:
        manager = PersonalityManager()
        instruction = manager.get_instruction("pt-BR")
        # Returns the Brazilian commentator personality
    """
    
    def __init__(self, custom_personalities: Optional[Dict[str, str]] = None):
        """
        Initialize the Personality Manager.
        
        Args:
            custom_personalities: Optional custom personalities to add/override
        """
        self.personalities = PERSONALITIES.copy()
        
        if custom_personalities:
            self.personalities.update(custom_personalities)
        
        self.default_locale = "en-US"
        logger.info(f"PersonalityManager initialized with {len(self.personalities)} personalities")
    
    def get_instruction(self, locale: str) -> str:
        """
        Get the system instruction for a specific locale.
        
        Args:
            locale: Locale code (e.g., 'en-US', 'pt-BR', 'en-GB')
            
        Returns:
            System instruction string for that personality
        """
        if locale in self.personalities:
            logger.debug(f"Loading personality for locale: {locale}")
            return self.personalities[locale]
        else:
            logger.warning(f"Unknown locale '{locale}', defaulting to {self.default_locale}")
            return self.personalities[self.default_locale]
    
    def inject_context(self, instruction: str, game_state: str) -> str:
        """
        Combine personality instruction with current game context.
        
        Args:
            instruction: Base personality instruction
            game_state: Current game state/context
            
        Returns:
            Combined instruction with context
        """
        return f"""{instruction}

CURRENT GAME STATE:
{game_state}

Respond to events in your personality style while referencing the game state.
"""
    
    def list_locales(self) -> list:
        """List all available locale codes."""
        return list(self.personalities.keys())
    
    def add_personality(self, locale: str, instruction: str) -> None:
        """Add or update a personality."""
        self.personalities[locale] = instruction
        logger.info(f"Added personality for locale: {locale}")
    
    def remove_personality(self, locale: str) -> bool:
        """Remove a personality (cannot remove default)."""
        if locale == self.default_locale:
            logger.warning("Cannot remove default personality")
            return False
        if locale in self.personalities:
            del self.personalities[locale]
            logger.info(f"Removed personality for locale: {locale}")
            return True
        return False
    
    def get_personality_preview(self, locale: str, max_chars: int = 100) -> str:
        """Get a preview of a personality instruction."""
        instruction = self.get_instruction(locale)
        if len(instruction) > max_chars:
            return instruction[:max_chars] + "..."
        return instruction


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_default_manager = None


def get_manager() -> PersonalityManager:
    """Get the default PersonalityManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PersonalityManager()
    return _default_manager


def get_personality(locale: str) -> str:
    """Quick access to get a personality instruction."""
    return get_manager().get_instruction(locale)
