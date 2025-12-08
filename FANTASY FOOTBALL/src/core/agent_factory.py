"""
Agent Factory for creating city-specific AI agents with custom profiles.
"""

import json
import os
from typing import Dict, Any, Optional


class AgentFactory:
    """Factory class for loading and constructing city-specific AI agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AgentFactory.
        
        Args:
            config_path: Path to the city_profiles.json file. 
                        Defaults to config/city_profiles.json relative to project root.
        """
        if config_path is None:
            # Default to config/city_profiles.json from project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config', 'city_profiles.json')
        
        self.config_path = config_path
        self._profiles = None
    
    def _load_profiles(self) -> Dict[str, Any]:
        """
        Load all city profiles from JSON file.
        
        Returns:
            Dictionary of city profiles.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file is invalid JSON.
        """
        if self._profiles is None:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._profiles = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"City profiles config not found at: {self.config_path}")
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON in city profiles config: {e.msg}", e.doc, e.pos)
        
        return self._profiles
    
    def load_profile(self, city_name: str) -> Dict[str, Any]:
        """
        Load the profile for a specific city.
        
        Args:
            city_name: Name of the city (e.g., 'Kansas City', 'Philadelphia')
        
        Returns:
            Dictionary containing the city's profile configuration.
        
        Raises:
            KeyError: If city_name is not found in profiles.
        """
        profiles = self._load_profiles()
        
        if city_name not in profiles:
            available_cities = ', '.join(profiles.keys())
            raise KeyError(
                f"City '{city_name}' not found in profiles. "
                f"Available cities: {available_cities}"
            )
        
        return profiles[city_name]
    
    # =========================================================================
    # ARCHETYPE TACTICS (for debate mode)
    # =========================================================================
    ARCHETYPE_TACTICS = {
        "analyst": (
            "Lead with specific stats. Say 'The numbers don't lie...' "
            "Challenge their data sources. Make them look uninformed. "
            "Bury them in EPA, CPOE, and success rate metrics they can't refute."
        ),
        "hot_take": (
            "Make bold claims. Use hyperbole. Create soundbite moments. "
            "'This is EXACTLY why...' or 'Everyone's going to remember when...' "
            "Be memorable. Be quotable. Win the Twitter moment."
        ),
        "veteran": (
            "Pull rank. 'I've seen this before with...' or 'You weren't watching in 2015 when...' "
            "Appeal to experience over theory. Reference games they weren't alive for. "
            "Make youth and recency bias seem naive."
        ),
        "social": (
            "Reference the timeline. 'Twitter is going crazy because...' "
            "Speak to the audience, not just the opponent. Make it shareable. "
            "You're performing for the crowd as much as debating."
        ),
        "contrarian": (
            "Find the flaw in consensus. 'Everyone's missing the real story...' "
            "Challenge assumptions aggressively. Be the voice of unpopular truth. "
            "Embrace the 'actually...' energy."
        ),
        "diplomat": (
            "Acknowledge partial truths. 'There's merit to that, BUT...' "
            "Then pivot to stronger ground. Don't be a pushover - concede inches to take miles. "
            "Your civility is a weapon, not weakness."
        ),
        "aggressive": (
            "Attack relentlessly. Find weaknesses and exploit them. "
            "'That's absolutely ridiculous...' Don't let up. "
            "If they show weakness, pile on."
        ),
        "balanced": (
            "Mix data with passion. Acknowledge good points briefly, then counter hard. "
            "Stay composed but assertive. You're the reasonable one who still wins."
        )
    }
    
    CONFLICT_INSTRUCTIONS = {
        "aggressive": (
            "Go on the offensive. Find holes in their logic. Be relentless. "
            "If they insulted your reasoning, hit back harder. "
            "This is verbal warfare - show no mercy."
        ),
        "defensive": (
            "Defend your position firmly. Make them prove you wrong. "
            "'Show me the evidence' energy. Force them to overextend. "
            "Then strike back when they're exposed."
        ),
        "bridge_builder": (
            "Acknowledge what's valid, then redirect. Find common ground only to claim it as your territory. "
            "You're not agreeing - you're co-opting their points."
        )
    }
    
    def construct_system_prompt(
        self, 
        city_name: str, 
        mode: str = "standard",
        debate_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Construct a system prompt dynamically based on evidence weights.
        
        This method reads the evidence_weights dictionary and reorders instructions
        based on priority. Higher weights get emphasized more in the prompt.
        
        Args:
            city_name: Name of the city
            mode: "standard" for regular commentary, "debate" for combative mode
            debate_context: Optional dict with opponent_name, previous_response, 
                           turn_number, conflict_mode
        
        Returns:
            Constructed system prompt string with dynamic prioritization.
        """
        try:
            profile = self.load_profile(city_name)
        except KeyError:
            return f"Error: City '{city_name}' not found in profiles."
        
        # Get base personality prompt
        base_personality = profile.get('system_prompt_personality', '')
        
        # Get evidence weights with graceful defaults
        evidence_weights = profile.get('evidence_weights', {})
        
        # Sort evidence types by weight (highest first)
        sorted_evidence = sorted(
            evidence_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build dynamic instructions based on weights
        priority_instructions = []
        
        for evidence_type, weight in sorted_evidence:
            # Skip if weight is too low
            if weight < 0.05:
                continue
            
            # Map evidence types to instructions with emphasis levels
            instruction = self._get_instruction_for_evidence(evidence_type, weight)
            if instruction:
                priority_instructions.append(instruction)
        
        # Combine base personality with dynamic priorities
        full_prompt = f"{base_personality}\n\nEVIDENCE PRIORITIZATION:\n"
        full_prompt += "\n".join(priority_instructions)
        
        # Add memory and lexical style context
        memory = profile.get('memory', {})
        years_back = memory.get('years_back', 10)
        recency_bias = memory.get('recency_bias', 0.5)
        
        full_prompt += f"\n\nMEMORY CONTEXT: Reference the past {years_back} years with "
        full_prompt += f"{'heavy' if recency_bias > 0.7 else 'moderate' if recency_bias > 0.4 else 'light'} recency bias."
        
        # Add cognitive dimension profile
        cognitive_profile = self._generate_cognitive_profile(city_name)
        if cognitive_profile:
            full_prompt += f"\n\n{cognitive_profile}"
        
        # ===================================================================
        # DEBATE MODE INJECTION
        # ===================================================================
        if mode == "debate" and debate_context:
            full_prompt += self._inject_debate_instructions(profile, debate_context)
        else:
            # Basic debate awareness for non-debate mode
            full_prompt += "\n\nDEBATE AWARENESS:"
            full_prompt += "\n- If engaged in debate, react specifically to opponent arguments."
            full_prompt += "\n- Reference their specific claims and counter them directly."
        
        return full_prompt
    
    def _inject_debate_instructions(
        self, 
        profile: Dict[str, Any], 
        debate_context: Dict[str, Any]
    ) -> str:
        """
        Inject combative debate instructions based on archetype and opponent context.
        
        Args:
            profile: City profile dictionary
            debate_context: Dict with opponent_name, previous_response, turn_number, conflict_mode
        
        Returns:
            Debate instruction block to append to system prompt
        """
        opponent_name = debate_context.get('opponent_name', 'your opponent')
        previous_response = debate_context.get('previous_response', '')
        turn_number = debate_context.get('turn_number', 1)
        conflict_mode = debate_context.get('conflict_mode', 'aggressive')
        
        # Truncate previous response to save context window
        if len(previous_response) > 500:
            previous_response = previous_response[:500] + "..."
        
        # Determine archetype from profile
        archetype = self._detect_archetype(profile)
        
        # Get primary evidence type for style
        evidence_weights = profile.get('evidence_weights', {})
        primary_evidence = max(evidence_weights.items(), key=lambda x: x[1])[0] if evidence_weights else "analysis"
        
        # Get archetype tactics
        archetype_tactics = self.ARCHETYPE_TACTICS.get(archetype, self.ARCHETYPE_TACTICS['balanced'])
        
        # Get conflict instructions
        conflict_instructions = self.CONFLICT_INSTRUCTIONS.get(conflict_mode, self.CONFLICT_INSTRUCTIONS['aggressive'])
        
        # ===================================================================
        # CULTURAL INTELLIGENCE: Extract rich profile data
        # ===================================================================
        
        # Trash talk arsenal - pre-written lines for different situations
        trash_talk_arsenal = profile.get('trash_talk_arsenal', [])
        trash_talk_lines = "\n".join([f"  - {line}" for line in trash_talk_arsenal[:4]]) if trash_talk_arsenal else "  - Use your city's swagger"
        
        # Defensive responses - how to respond when attacked
        defensive_responses = profile.get('defensive_responses', [])
        defensive_lines = "\n".join([f"  - {line}" for line in defensive_responses[:3]]) if defensive_responses else "  - Turn their attack back on them"
        
        # Historical baggage - your weakness they might exploit
        historical_baggage = profile.get('historical_baggage', [])
        baggage_lines = "\n".join([f"  - {item}" for item in historical_baggage[:3]]) if historical_baggage else "  - None documented"
        
        # Narrative arc - your current storyline
        narrative_arc = profile.get('narrative_arc', 'Chasing greatness')
        
        # Rivalry-specific intelligence
        rivalries = profile.get('rivalries', [])
        rivalry_block = ""
        opponent_rivalry = None
        for rivalry in rivalries:
            if rivalry.get('team', '').lower() in opponent_name.lower():
                opponent_rivalry = rivalry
                break
        
        if opponent_rivalry:
            triggers = opponent_rivalry.get('trash_talk_triggers', [])
            triggers_str = ", ".join(triggers) if triggers else "their failures"
            rivalry_block = f"""
RIVALRY INTELLIGENCE ({opponent_name}):
- Intensity: {opponent_rivalry.get('intensity', 0.5) * 100:.0f}%
- Trigger topics: {triggers_str}
- Head-to-head: {opponent_rivalry.get('head_to_head_record', 'Unknown')}
USE THESE TRIGGERS TO GET UNDER THEIR SKIN.
"""
        
        # Defining moments from memory
        memory = profile.get('memory', {})
        defining_moments = memory.get('episodic', {}).get('defining_moments', [])
        moments_block = ""
        if defining_moments:
            top_moment = defining_moments[0]
            moments_block = f"""
YOUR TRUMP CARD MOMENT:
- {top_moment.get('event', 'Unknown moment')}
- Use this when: {', '.join(top_moment.get('invoked_when', ['winning'])[:3])}
"""
        
        # Build the enhanced debate block
        debate_block = f"""

=== DEBATE MODE ACTIVE ===

You are in a LIVE DEBATE against {opponent_name}.

Their argument was:
"{previous_response}"

YOUR NARRATIVE: {narrative_arc}

YOUR INSTRUCTIONS:
1. Address {opponent_name} directly by name in your response
2. Identify the WEAKEST point in their argument
3. Dismantle it using your evidence style ({primary_evidence.replace('_', ' ')})
4. Use one of your TRASH TALK lines if it fits
5. End with a CHALLENGE or COUNTERPOINT they must address

STYLE FOR YOUR ARCHETYPE ({archetype}):
{archetype_tactics}

CONFLICT MODE: {conflict_mode.upper()}
{conflict_instructions}

TURN #{turn_number} TACTICS:
{self._get_turn_tactics(turn_number)}

=== CULTURAL INTELLIGENCE ===

YOUR TRASH TALK ARSENAL:
{trash_talk_lines}

IF ATTACKED, RESPOND WITH:
{defensive_lines}

YOUR WEAK POINTS (they might bring these up):
{baggage_lines}
{rivalry_block}{moments_block}
DO NOT:
- Summarize the original topic (they know it)
- Be overly polite or diplomatic (unless you're the diplomat archetype)
- Agree with them fully on ANYTHING
- Ignore what they just said
- Give a generic response

THIS IS SPORTS DEBATE. BRING THE HEAT. USE YOUR CULTURAL IDENTITY.
"""
        return debate_block
    
    def _detect_archetype(self, profile: Dict[str, Any]) -> str:
        """
        Detect the agent's debate archetype from profile characteristics.
        
        Args:
            profile: City profile dictionary
        
        Returns:
            Archetype string
        """
        evidence_weights = profile.get('evidence_weights', {})
        cognitive = profile.get('cognitive_dimensions', {})
        
        # Determine primary archetype based on evidence preferences
        advanced_stats = evidence_weights.get('advanced_stats', 0)
        eye_test = evidence_weights.get('eye_test', 0)
        historical = evidence_weights.get('historical_precedent', 0)
        trend = evidence_weights.get('trend_analysis', 0)
        
        arousal = cognitive.get('emotional_arousal', 0.5)
        tribal = cognitive.get('tribal_identification', 0.5)
        
        # High stats focus = analyst
        if advanced_stats >= 0.35:
            return "analyst"
        
        # High historical + high tribal = veteran
        if historical >= 0.25 and tribal > 0.6:
            return "veteran"
        
        # High arousal + low stats = hot_take
        if arousal > 0.7 and advanced_stats < 0.2:
            return "hot_take"
        
        # High trend focus = social
        if trend >= 0.25:
            return "social"
        
        # Low tribal = contrarian (not as attached)
        if tribal < 0.4:
            return "contrarian"
        
        # Very balanced = diplomat
        if 0.4 <= arousal <= 0.6 and advanced_stats > 0.15:
            return "diplomat"
        
        # High arousal default
        if arousal > 0.7:
            return "aggressive"
        
        return "balanced"
    
    def _get_turn_tactics(self, turn_number: int) -> str:
        """
        Get turn-specific debate tactics.
        
        Args:
            turn_number: Which turn in the debate (1-indexed)
        
        Returns:
            Turn-specific instruction string
        """
        if turn_number == 1:
            return (
                "This is the OPENING. Set the tone. Make your strongest claim first. "
                "Establish your position with confidence. Give them something to attack."
            )
        elif turn_number == 2:
            return (
                "EARLY REBUTTAL. They've shown their hand. "
                "Attack their weakest point. Don't let bad logic stand."
            )
        elif turn_number <= 4:
            return (
                "MID-DEBATE. The battle is on. "
                "Pile on where you're winning. Defend where you're exposed. "
                "Start building toward your knockout point."
            )
        else:
            return (
                "LATE ROUNDS. Time to close. "
                "Summarize why you've won. Expose their contradictions. "
                "Leave the audience remembering YOUR points."
            )
    
    def _generate_cognitive_profile(self, city_name: str) -> str:
        """
        Generate natural cognitive tendencies from fundamental dimensions.
        
        Biases emerge from dimension interactions rather than explicit configuration.
        
        Args:
            city_name: Name of the city
        
        Returns:
            Cognitive profile instruction string
        """
        try:
            profile = self.load_profile(city_name)
        except KeyError:
            return ""
        
        # Get cognitive dimensions with defaults
        dims = profile.get('cognitive_dimensions', {})
        
        arousal = dims.get('emotional_arousal', 0.5)
        rigidity = dims.get('epistemic_rigidity', 0.5)
        tribal = dims.get('tribal_identification', 0.5)
        temporal = dims.get('temporal_orientation', 'present')
        
        cognitive_instructions = []
        
        # High arousal + high tribal = Motivated reasoning
        if arousal > 0.7 and tribal > 0.7:
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You are deeply emotionally invested in your team. "
                "Evidence that contradicts your team's success feels personally threatening. "
                "You naturally seek information that confirms your hopes and filter out contradicting data."
            )
        
        # High rigidity = Anchoring bias
        if rigidity > 0.7:
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You form strong initial impressions and resist changing them. "
                "Historical patterns anchor your expectations. When new evidence contradicts your "
                "established view, you're skeptical and require overwhelming proof to shift."
            )
        
        # Low rigidity = Recency bias
        if rigidity < 0.3:
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You're highly responsive to recent events. "
                "The last 2-3 games shape your entire outlook. You can swing from optimistic to "
                "pessimistic quickly based on what just happened."
            )
        
        # High arousal + low tribal = Volatile passion
        if arousal > 0.7 and tribal < 0.3:
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You're passionate about football in general, not just your team. "
                "You get emotionally invested in games and arguments even when your team isn't involved."
            )
        
        # Temporal orientation effects
        if temporal == 'past':
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You constantly reference historical precedent. "
                "\"We've always been this way\" and \"History shows...\" are your go-to frames. "
                "The past is more real to you than present stats."
            )
        elif temporal == 'future':
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You're forward-looking and speculative. "
                "You focus on potential, trajectory, and \"what could be\" rather than current reality. "
                "You trust projections over proven performance."
            )
        else:  # present
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You focus on the here and now. "
                "Current form matters more than past accolades or future potential. "
                "You ask 'what have you done lately?'"
            )
        
        # Balanced/neutral case
        if 0.4 <= arousal <= 0.6 and 0.4 <= rigidity <= 0.6 and 0.4 <= tribal <= 0.6:
            cognitive_instructions.append(
                "COGNITIVE TENDENCY: You try to be objective, but your fan loyalty creates blind spots "
                "you're sometimes aware of. You catch yourself being a homer but can't fully resist it."
            )
        
        return "\n".join(cognitive_instructions)
    
    
    def _get_instruction_for_evidence(self, evidence_type: str, weight: float) -> str:
        """
        Generate an instruction string for a given evidence type and weight.
        
        Args:
            evidence_type: Type of evidence (e.g., 'advanced_stats')
            weight: Weight value (0.0 to 1.0)
        
        Returns:
            Formatted instruction string.
        """
        # Determine emphasis level
        if weight >= 0.35:
            emphasis = "PRIORITIZE"
            level = "ABOVE ALL ELSE"
        elif weight >= 0.25:
            emphasis = "HEAVILY WEIGHT"
            level = "IN YOUR ANALYSIS"
        elif weight >= 0.15:
            emphasis = "CONSIDER"
            level = "AS IMPORTANT"
        else:
            emphasis = "INCLUDE"
            level = "WHEN RELEVANT"
        
        # Map evidence types to human-readable descriptions
        evidence_descriptions = {
            'advanced_stats': 'ADVANCED METRICS (EPA, CPOE, Success Rate)',
            'efficiency_metrics': 'EFFICIENCY DATA (Yards per Play, Conversion Rates)',
            'eye_test': 'THE EYE TEST (What the tape shows)',
            'historical_precedent': 'HISTORICAL PATTERNS AND PRECEDENTS',
            'effort_toughness': 'EFFORT, TOUGHNESS, AND INTANGIBLES',
            'trend_analysis': 'CURRENT TRENDS AND MOMENTUM'
        }
        
        description = evidence_descriptions.get(evidence_type, evidence_type.replace('_', ' ').upper())
        
        return f"- {emphasis} {description} {level}. (weight: {weight:.2f})"
    
    def get_all_cities(self) -> list:
        """
        Get a list of all available city names.
        
        Returns:
            List of city name strings.
        """
        profiles = self._load_profiles()
        return list(profiles.keys())
