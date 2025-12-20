"""
learning_agent.py - LearningAgent for neuron_core

Agent that learns from experience to improve performance.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, capability
from ..types import AgentID, Message

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """
    Agent that learns from experience to improve performance.
    
    Updates its behavior based on feedback and experience.
    """
    
    def __init__(self, agent_id: Optional[AgentID] = None, name: str = "",
                 description: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id,
            name or "LearningAgent",
            description or "Agent that learns from experience",
            metadata
        )
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.discount_factor = 0.9
        self.knowledge_base: Dict[str, Any] = {}
        self.learned_patterns: Dict[str, Any] = {}
        self.experience_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def _initialize(self) -> None:
        """Initialize learning model."""
        self._setup_learning_model()
    
    def _setup_learning_model(self) -> None:
        """Setup the learning model."""
        self.knowledge_base = {
            "concepts": {},
            "patterns": {},
            "feedback": {}
        }
    
    def set_learning_parameters(self, learning_rate: float, 
                               exploration_rate: float,
                               discount_factor: float) -> None:
        """Set learning parameters."""
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("Exploration rate must be between 0.0 and 1.0")
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("Discount factor must be between 0.0 and 1.0")
        
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
    
    def add_to_knowledge_base(self, concept: str, data: Any) -> None:
        """Add information to the knowledge base."""
        if concept not in self.knowledge_base["concepts"]:
            self.knowledge_base["concepts"][concept] = []
        self.knowledge_base["concepts"][concept].append(data)
    
    def record_experience(self, experience: Dict[str, Any]) -> None:
        """Record an experience for learning."""
        if "timestamp" not in experience:
            experience["timestamp"] = time.time()
        
        self.experience_history.append(experience)
        
        if len(self.experience_history) > self.max_history_size:
            self.experience_history = self.experience_history[-self.max_history_size:]
        
        self._learn_from_experience(experience)
    
    def _learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from a single experience."""
        if "pattern_key" in experience and "pattern_value" in experience:
            key = experience["pattern_key"]
            value = experience["pattern_value"]
            
            if key not in self.learned_patterns:
                self.learned_patterns[key] = {"values": [], "counts": {}}
            
            self.learned_patterns[key]["values"].append(value)
            
            if value not in self.learned_patterns[key]["counts"]:
                self.learned_patterns[key]["counts"][value] = 0
            self.learned_patterns[key]["counts"][value] += 1
    
    def receive_feedback(self, feedback_id: str, score: float, 
                        metadata: Dict[str, Any]) -> None:
        """Receive feedback on agent performance."""
        if not -1.0 <= score <= 1.0:
            raise ValueError("Feedback score must be between -1.0 and 1.0")
        
        if feedback_id not in self.knowledge_base["feedback"]:
            self.knowledge_base["feedback"][feedback_id] = []
        
        self.knowledge_base["feedback"][feedback_id].append({
            "score": score,
            "timestamp": time.time(),
            "metadata": metadata
        })
    
    @capability(
        name="learning_process",
        description="Process input with learning capability",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    async def process_message(self, message: Message) -> None:
        """Process a message with learning."""
        if message.sender == self.id:
            return
        
        # Record the experience
        self.record_experience({
            "type": "message_received",
            "content": message.content,
            "sender": message.sender
        })
        
        # Generate response based on learned patterns
        response = {
            "acknowledged": True,
            "patterns_known": len(self.learned_patterns),
            "learning_rate": self.learning_rate
        }
        
        await self.send_message(
            recipients=message.sender,
            content=response,
            metadata={"in_response_to": message.id, "reasoning_path": "learning"}
        )
