"""
Memory Module for Cultural Cognition System.

Implements tiered memory architecture:
- Episodic: Specific events with emotional weight
- Semantic: Team archetypes and player narratives
- Procedural: Learned argument patterns
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EpisodicMemory:
    """A specific memorable event."""
    event: str
    emotional_weight: float  # 0.0 to 1.0
    invoked_when: List[str]  # Context triggers
    timestamp: Optional[str] = None


@dataclass
class SemanticKnowledge:
    """General knowledge about teams/players."""
    entity: str
    archetype: str
    description: str


class MemoryModule:
    """Handles tiered memory retrieval for cultural agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize memory module.
        
        Args:
            config_path: Path to city_profiles.json
        """
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config', 'city_profiles.json')
        
        self.config_path = config_path
        self._profiles = None
    
    def _load_profiles(self) -> Dict[str, Any]:
        """Load city profiles from JSON."""
        if self._profiles is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._profiles = json.load(f)
        return self._profiles
    
    def _load_city_profile(self, city_name: str) -> Dict[str, Any]:
        """Load a specific city's profile."""
        profiles = self._load_profiles()
        if city_name not in profiles:
            raise KeyError(f"City '{city_name}' not found in profiles")
        return profiles[city_name]
    
    def get_episodic_memories(
        self, 
        city_name: str, 
        context_triggers: Optional[List[str]] = None
    ) -> List[EpisodicMemory]:
        """
        Retrieve episodic memories for a city.
        
        Args:
            city_name: Name of the city
            context_triggers: Optional list of context keywords to filter by
        
        Returns:
            List of relevant episodic memories, sorted by emotional weight
        """
        try:
            profile = self._load_city_profile(city_name)
            memory_config = profile.get('memory', {})
            episodic_data = memory_config.get('episodic', {})
            
            # Parse defining moments
            defining_moments = episodic_data.get('defining_moments', [])
            memories = []
            
            for moment in defining_moments:
                memory = EpisodicMemory(
                    event=moment.get('event', ''),
                    emotional_weight=moment.get('emotional_weight', 0.5),
                    invoked_when=moment.get('invoked_when', []),
                    timestamp=moment.get('timestamp')
                )
                
                # Filter by context triggers if provided
                if context_triggers:
                    # Check if any trigger matches
                    if any(trigger in memory.invoked_when for trigger in context_triggers):
                        memories.append(memory)
                else:
                    memories.append(memory)
            
            # Sort by emotional weight (highest first)
            memories.sort(key=lambda m: m.emotional_weight, reverse=True)
            
            return memories
            
        except KeyError:
            # City not found or no episodic memory configured
            return []
    
    def get_semantic_knowledge(
        self, 
        city_name: str, 
        entity_filter: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Retrieve semantic knowledge (archetypes, narratives).
        
        Args:
            city_name: Name of the city
            entity_filter: Optional entity name to filter (e.g., "Cowboys")
        
        Returns:
            Dictionary of entity -> archetype mappings
        """
        try:
            profile = self._load_city_profile(city_name)
            memory_config = profile.get('memory', {})
            semantic_data = memory_config.get('semantic', {})
            
            team_archetypes = semantic_data.get('team_archetypes', {})
            player_narratives = semantic_data.get('player_narratives', {})
            
            # Combine both
            all_knowledge = {**team_archetypes, **player_narratives}
            
            if entity_filter:
                # Return only the specific entity
                return {entity_filter: all_knowledge.get(entity_filter, 'unknown')}
            
            return all_knowledge
            
        except KeyError:
            return {}
    
    def get_procedural_patterns(self, city_name: str) -> List[str]:
        """
        Retrieve learned argument patterns.
        
        Args:
            city_name: Name of the city
        
        Returns:
            List of argument pattern strings
        """
        try:
            profile = self._load_city_profile(city_name)
            memory_config = profile.get('memory', {})
            procedural_data = memory_config.get('procedural', {})
            
            return procedural_data.get('argument_patterns', [])
            
        except KeyError:
            return []
    
    def construct_memory_context(
        self, 
        city_name: str, 
        current_situation: Dict[str, Any]
    ) -> str:
        """
        Construct a memory-enhanced context string for prompts.
        
        Args:
            city_name: Name of the city
            current_situation: Current conversation context
                {
                    'opponent_team': str,
                    'topic': str,
                    'keywords': List[str]
                }
        
        Returns:
            Formatted memory context string
        """
        context_parts = []
        
        # Extract context triggers from situation
        keywords = current_situation.get('keywords', [])
        opponent = current_situation.get('opponent_team')
        
        if opponent:
            keywords.append(opponent)
        
        # Get episodic memories
        episodic_memories = self.get_episodic_memories(city_name, keywords)
        if episodic_memories:
            context_parts.append("DEFINING MOMENTS YOU REMEMBER:")
            for memory in episodic_memories[:3]:  # Top 3 most relevant
                weight_desc = "deeply" if memory.emotional_weight > 0.8 else "significantly" if memory.emotional_weight > 0.5 else "somewhat"
                context_parts.append(f"- {memory.event} (affects you {weight_desc})")
        
        # Get semantic knowledge about opponent/players
        semantic_knowledge = self.get_semantic_knowledge(city_name, opponent)
        if semantic_knowledge:
            context_parts.append("\nHOW YOU VIEW TEAMS/PLAYERS:")
            for entity, archetype in semantic_knowledge.items():
                context_parts.append(f"- {entity}: {archetype}")
        
        # Get procedural patterns
        patterns = self.get_procedural_patterns(city_name)
        if patterns:
            context_parts.append("\nYOUR NATURAL ARGUMENT TENDENCIES:")
            for pattern in patterns[:2]:  # Top 2 patterns
                context_parts.append(f"- {pattern}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def calculate_memory_relevance_score(
        self, 
        memory: EpisodicMemory, 
        current_context: Dict[str, Any]
    ) -> float:
        """
        Calculate how relevant a memory is to the current context.
        
        Args:
            memory: The episodic memory to score
            current_context: Current conversation context
        
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        keywords = current_context.get('keywords', [])
        
        # Check keyword matches
        for keyword in keywords:
            if keyword.lower() in ' '.join(memory.invoked_when).lower():
                score += 0.3
        
        # Factor in emotional weight
        score += memory.emotional_weight * 0.4
        
        # Recent memories get slight boost (if timestamp available)
        # TODO: Implement temporal weighting
        
        return min(score, 1.0)
