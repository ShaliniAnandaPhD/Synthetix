"""
Live Agent Dispatcher for Live Commentary

Routes classified events to the appropriate agent types based on urgency.
Manages parallel agent spawning via Modal with appropriate latency budgets.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Callable
import time
import logging

from .event_classifier import ClassifiedEvent, EventUrgency

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from a commentary agent"""
    region: str
    agent_type: str
    text: str
    emotion: str
    confidence: float
    latency_ms: int
    voice_id: Optional[str] = None
    priority: int = 5  # 1-10, higher = speak sooner
    
    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "agent_type": self.agent_type,
            "text": self.text,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "voice_id": self.voice_id,
            "priority": self.priority
        }


class LiveAgentDispatcher:
    """
    Dispatches events to agents based on urgency and type.
    Manages parallel agent activation for live commentary.
    
    Usage:
        dispatcher = LiveAgentDispatcher(modal_app, regional_config)
        async for response in dispatcher.dispatch(event, ["dallas", "kansas_city"]):
            print(f"{response.region}: {response.text}")
    """
    
    # Agent types for each urgency level
    URGENCY_AGENTS = {
        EventUrgency.IMMEDIATE: ["homer", "analyst"],  # Fast reactions
        EventUrgency.STRATEGIC: ["homer", "analyst", "contrarian", "historian"],  # Full panel
        EventUrgency.CONTEXTUAL: ["historian", "stats_expert"],  # Fill time
    }
    
    # Priority by agent type (higher = speaks first)
    AGENT_PRIORITY = {
        "homer": 9,       # React first with emotion
        "analyst": 7,     # Quick data take
        "contrarian": 5,  # Counter-point
        "historian": 4,   # Context
        "stats_expert": 3,
    }
    
    # Voice IDs by region (from your existing config)
    REGIONAL_VOICES = {
        "dallas": "JBFqnCBsd6RMkjVDRZzb",
        "kansas_city": "21m00Tcm4TlvDq8ikWAM",
        "philadelphia": "AZnzlk1XvdvUeBnXmlld",
        "new_york": "ErXwobaYiN019PkySvjV",
        "chicago": "VR6AewLTigWG4xSOukaG",
        "green_bay": "pNInz6obpgDQGcFmaJgB",
        "new_england": "ODq5zmih8GrVes37Dizd",
        "san_francisco": "ThT5KcBeYPX3keUQqHPh",
        "default": "onwK4e9ZLuTAKqWW03F9",
    }
    
    def __init__(
        self, 
        modal_app=None, 
        regional_config: Optional[dict] = None,
        reaction_fn: Optional[Callable] = None,
        analysis_fn: Optional[Callable] = None
    ):
        """
        Initialize dispatcher.
        
        Args:
            modal_app: Modal app instance for remote calls
            regional_config: Dict of region -> cultural parameters
            reaction_fn: Optional custom function for reaction agents
            analysis_fn: Optional custom function for analysis agents
        """
        self.modal_app = modal_app
        self.regional_config = regional_config or {}
        self.reaction_fn = reaction_fn
        self.analysis_fn = analysis_fn
        self.active_agents = {}
        self._response_count = 0
    
    async def dispatch(
        self, 
        event: ClassifiedEvent,
        creator_regions: list[str]
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Dispatch event to appropriate agents, yield responses as they arrive.
        
        Args:
            event: The classified event to process
            creator_regions: List of regions for this creator's panel
        
        Yields:
            AgentResponse objects as agents complete
        """
        start_time = time.time()
        
        # Determine which agent types to activate
        agent_types = self._select_agent_types(event)
        logger.info(f"Dispatching {event.event_type.value} to agents: {agent_types} for regions: {creator_regions}")
        
        # Spawn agents in parallel
        tasks = []
        for region in creator_regions:
            for agent_type in agent_types:
                task = asyncio.create_task(
                    self._spawn_agent(event, region, agent_type)
                )
                tasks.append((region, agent_type, task))
        
        # Yield responses as they complete (don't wait for all)
        timeout_seconds = event.latency_budget_ms / 1000
        
        for region, agent_type, task in tasks:
            try:
                response = await asyncio.wait_for(
                    task, 
                    timeout=timeout_seconds
                )
                if response:
                    self._response_count += 1
                    yield response
            except asyncio.TimeoutError:
                logger.warning(f"Agent {region}/{agent_type} timed out after {event.latency_budget_ms}ms")
                # For live commentary, we don't wait - skip this agent
                continue
            except Exception as e:
                logger.error(f"Agent {region}/{agent_type} error: {e}")
                continue
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Dispatch complete in {total_time:.0f}ms, got {self._response_count} responses")
    
    def _select_agent_types(self, event: ClassifiedEvent) -> list[str]:
        """Select which agent archetypes should respond based on urgency"""
        return self.URGENCY_AGENTS.get(event.urgency, ["analyst"])
    
    async def _spawn_agent(
        self, 
        event: ClassifiedEvent, 
        region: str, 
        agent_type: str
    ) -> Optional[AgentResponse]:
        """
        Spawn a single agent to respond to the event.
        
        In production, this calls Modal functions.
        For testing, uses mock responses.
        """
        start = time.time()
        
        # Get regional parameters
        params = self.regional_config.get(region, {})
        voice_id = self.REGIONAL_VOICES.get(region, self.REGIONAL_VOICES["default"])
        
        try:
            if event.urgency == EventUrgency.IMMEDIATE:
                # Use lightweight, fast reaction
                response_data = await self._run_reaction_agent(
                    event, region, agent_type, params
                )
            else:
                # Use heavier analysis
                response_data = await self._run_analysis_agent(
                    event, region, agent_type, params
                )
            
            latency = int((time.time() - start) * 1000)
            
            return AgentResponse(
                region=region,
                agent_type=agent_type,
                text=response_data.get("text", ""),
                emotion=response_data.get("emotion", "neutral"),
                confidence=response_data.get("confidence", 0.8),
                latency_ms=latency,
                voice_id=voice_id,
                priority=self.AGENT_PRIORITY.get(agent_type, 5)
            )
            
        except Exception as e:
            logger.error(f"Agent spawn failed for {region}/{agent_type}: {e}")
            return None
    
    async def _run_reaction_agent(
        self, 
        event: ClassifiedEvent, 
        region: str, 
        agent_type: str,
        params: dict
    ) -> dict:
        """
        Run fast reaction agent (sub-150ms target).
        """
        if self.reaction_fn:
            return await self.reaction_fn(event, region, agent_type, params)
        
        if self.modal_app:
            # Call Modal function
            return await self.modal_app.functions.run_reaction_agent.remote(
                event=event.to_dict(),
                region=region,
                agent_type=agent_type,
                params=params
            )
        
        # Mock response for testing
        return self._mock_reaction(event, region, agent_type)
    
    async def _run_analysis_agent(
        self, 
        event: ClassifiedEvent, 
        region: str, 
        agent_type: str,
        params: dict
    ) -> dict:
        """
        Run analysis agent (sub-500ms target).
        """
        if self.analysis_fn:
            return await self.analysis_fn(event, region, agent_type, params)
        
        if self.modal_app:
            return await self.modal_app.functions.run_analysis_agent.remote(
                event=event.to_dict(),
                region=region,
                agent_type=agent_type,
                params=params
            )
        
        # Mock response for testing
        return self._mock_analysis(event, region, agent_type)
    
    def _mock_reaction(self, event: ClassifiedEvent, region: str, agent_type: str) -> dict:
        """Generate mock reaction for testing"""
        reactions = {
            "homer": {
                "touchdown": f"TOUCHDOWN! That's what I'm talking about! {region.replace('_', ' ').title()} style!",
                "interception": f"Oh no! You hate to see it but that's the game sometimes!",
                "big_play": f"WHAT A PLAY! This is why we watch!",
            },
            "analyst": {
                "touchdown": f"Excellent execution on that drive. Six points on the board.",
                "interception": f"That's a costly turnover. Momentum shift coming.",
                "big_play": f"Great scheme recognition there. Took advantage of the coverage.",
            }
        }
        
        event_key = event.event_type.value
        agent_reactions = reactions.get(agent_type, reactions["analyst"])
        text = agent_reactions.get(event_key, f"Interesting development here from {region}.")
        
        return {
            "text": text,
            "emotion": "excited" if event.urgency == EventUrgency.IMMEDIATE else "analytical",
            "confidence": 0.85
        }
    
    def _mock_analysis(self, event: ClassifiedEvent, region: str, agent_type: str) -> dict:
        """Generate mock analysis for testing"""
        analyses = {
            "historian": f"Historically, {region.replace('_', ' ').title()} has seen this scenario before...",
            "contrarian": f"I'm going to push back on the conventional wisdom here...",
            "stats_expert": f"The numbers tell an interesting story in this situation...",
        }
        
        text = analyses.get(agent_type, f"From a {region} perspective, this is notable.")
        
        return {
            "text": text,
            "emotion": "thoughtful",
            "confidence": 0.75
        }


# Factory function
def create_dispatcher(modal_app=None, regional_config: dict = None) -> LiveAgentDispatcher:
    """Create a configured dispatcher instance"""
    return LiveAgentDispatcher(
        modal_app=modal_app,
        regional_config=regional_config or {}
    )


if __name__ == "__main__":
    # Test the dispatcher
    import asyncio
    from .event_classifier import EventClassifier
    
    async def test():
        classifier = EventClassifier()
        dispatcher = LiveAgentDispatcher()
        
        event = classifier.classify({
            "description": "Patrick Mahomes TOUCHDOWN pass to Travis Kelce!",
            "type": "play"
        })
        
        print(f"\nEvent: {event.event_type.value}, Urgency: {event.urgency.value}")
        print("Dispatching to agents...")
        
        async for response in dispatcher.dispatch(event, ["kansas_city", "dallas"]):
            print(f"\n{response.region}/{response.agent_type}:")
            print(f"  {response.text}")
            print(f"  Emotion: {response.emotion}, Latency: {response.latency_ms}ms")
    
    asyncio.run(test())
