"""
deliberative_agent.py - DeliberativeAgent for neuron_core

Agent that performs deliberative reasoning to process messages.
Inspired by the brain's prefrontal cortex.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, capability
from ..types import AgentID, Message

logger = logging.getLogger(__name__)


class DeliberativeAgent(BaseAgent):
    """
    Agent that uses deliberative reasoning to process messages.
    
    Uses model-based reasoning to evaluate options and plan responses,
    allowing for more complex decision-making than reflex agents.
    """
    
    def __init__(self, agent_id: Optional[AgentID] = None, name: str = "",
                 description: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id,
            name or "DeliberativeAgent",
            description or "Agent that uses deliberative reasoning",
            metadata
        )
        self.reasoning_model = None
        self.planning_depth = 3
        self.evaluation_criteria: Dict[str, float] = {}
    
    def _initialize(self) -> None:
        """Initialize reasoning model and evaluation criteria."""
        self._setup_reasoning_model()
        if not self.evaluation_criteria:
            self.evaluation_criteria = self._get_default_evaluation_criteria()
    
    def _setup_reasoning_model(self) -> None:
        """Setup the reasoning model."""
        self.reasoning_model = {
            "process_steps": [
                "understand_request",
                "generate_options",
                "evaluate_options",
                "select_best_option",
                "execute_action"
            ]
        }
    
    def _get_default_evaluation_criteria(self) -> Dict[str, float]:
        """Get default criteria for evaluating options."""
        return {
            "efficiency": 0.3,
            "completeness": 0.3,
            "reliability": 0.2,
            "innovation": 0.1,
            "simplicity": 0.1
        }
    
    def set_evaluation_criteria(self, criteria: Dict[str, float]) -> None:
        """
        Set criteria for evaluating options.
        
        Args:
            criteria: Dictionary mapping criteria names to weights (must sum to 1.0)
        """
        total_weight = sum(criteria.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        self.evaluation_criteria = criteria
    
    def set_planning_depth(self, depth: int) -> None:
        """Set planning depth (steps ahead to consider)."""
        if depth < 1:
            raise ValueError("Planning depth must be at least 1")
        self.planning_depth = depth
    
    @capability(
        name="deliberative_reasoning",
        description="Process input using deliberative reasoning",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    async def process_message(self, message: Message) -> None:
        """
        Process a message using deliberative reasoning.
        
        Steps:
        1. Understand the request
        2. Generate options
        3. Evaluate options
        4. Select the best option
        5. Execute the action
        """
        if message.sender == self.id:
            return
        
        request = self._understand_request(message)
        options = self._generate_options(request)
        evaluated_options = self._evaluate_options(options, request)
        best_option = self._select_best_option(evaluated_options)
        result = self._execute_action(best_option, request)
        
        await self.send_message(
            recipients=message.sender,
            content=result,
            metadata={"in_response_to": message.id, "reasoning_path": "deliberative"}
        )
    
    def _understand_request(self, message: Message) -> Dict[str, Any]:
        """Understand the request in the message."""
        return {
            "original_message": message,
            "sender": message.sender,
            "timestamp": message.created_at,
            "understood_content": message.content
        }
    
    def _generate_options(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate options for responding."""
        return [
            {"type": "direct_response", "content": request["understood_content"]},
            {"type": "elaborated_response", "content": f"Elaborated: {request['understood_content']}"},
            {"type": "minimal_response", "content": "Acknowledged"}
        ]
    
    def _evaluate_options(self, options: List[Dict[str, Any]], 
                         request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate options against criteria."""
        evaluated_options = []
        
        for option in options:
            scores = {
                "efficiency": 0.5,
                "completeness": 0.5,
                "reliability": 0.5,
                "innovation": 0.5,
                "simplicity": 0.5
            }
            
            if option["type"] == "direct_response":
                scores["efficiency"] = 0.8
                scores["simplicity"] = 0.9
            elif option["type"] == "elaborated_response":
                scores["completeness"] = 0.9
                scores["innovation"] = 0.7
            elif option["type"] == "minimal_response":
                scores["efficiency"] = 0.9
                scores["simplicity"] = 1.0
                scores["completeness"] = 0.2
            
            total_score = sum(
                scores.get(criterion, 0.5) * weight
                for criterion, weight in self.evaluation_criteria.items()
            )
            
            evaluated_options.append({
                **option,
                "scores": scores,
                "total_score": total_score
            })
        
        return evaluated_options
    
    def _select_best_option(self, evaluated_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best option based on evaluation."""
        return max(evaluated_options, key=lambda x: x["total_score"])
    
    def _execute_action(self, option: Dict[str, Any], 
                       request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected action."""
        return {
            "type": option["type"],
            "content": option["content"],
            "confidence": option["total_score"],
            "request_understood": True
        }
