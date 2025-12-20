"""
coordinator_agent.py - CoordinatorAgent for neuron_core

Agent that coordinates and orchestrates other agents.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, capability
from ..types import AgentID, Message

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Agent that coordinates and orchestrates other agents.
    
    Manages workflows and delegates tasks to specialized agents.
    """
    
    def __init__(self, agent_id: Optional[AgentID] = None, name: str = "",
                 description: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id,
            name or "CoordinatorAgent",
            description or "Agent that coordinates other agents",
            metadata
        )
        self.managed_agents: Dict[AgentID, Dict[str, Any]] = {}
        self.workflows: Dict[str, List[Dict[str, Any]]] = {}
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
    
    def _initialize(self) -> None:
        """Initialize coordinator."""
        pass
    
    def register_managed_agent(self, agent_id: AgentID, 
                               capabilities: List[str],
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent to be managed by this coordinator."""
        self.managed_agents[agent_id] = {
            "capabilities": capabilities,
            "status": "ready",
            "metadata": metadata or {}
        }
        logger.info(f"Registered agent {agent_id} with coordinator {self.id}")
    
    def unregister_managed_agent(self, agent_id: AgentID) -> None:
        """Unregister a managed agent."""
        if agent_id in self.managed_agents:
            del self.managed_agents[agent_id]
    
    def define_workflow(self, workflow_name: str, 
                       steps: List[Dict[str, Any]]) -> None:
        """Define a workflow with multiple steps."""
        self.workflows[workflow_name] = steps
    
    def find_agent_for_capability(self, capability: str) -> Optional[AgentID]:
        """Find an agent that has a specific capability."""
        for agent_id, info in self.managed_agents.items():
            if capability in info["capabilities"] and info["status"] == "ready":
                return agent_id
        return None
    
    @capability(
        name="coordinate",
        description="Coordinate tasks across managed agents",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    async def process_message(self, message: Message) -> None:
        """Process a coordination request."""
        if message.sender == self.id:
            return
        
        content = message.content
        
        # Handle different coordination requests
        if isinstance(content, dict):
            action = content.get("action", "delegate")
            
            if action == "delegate":
                capability_needed = content.get("capability")
                if capability_needed:
                    target_agent = self.find_agent_for_capability(capability_needed)
                    if target_agent:
                        await self.send_message(
                            recipients=target_agent,
                            content=content.get("task"),
                            metadata={"delegated_from": self.id}
                        )
                        return
            
            elif action == "execute_workflow":
                workflow_name = content.get("workflow")
                if workflow_name in self.workflows:
                    # Execute workflow (simplified)
                    response = {
                        "workflow": workflow_name,
                        "steps": len(self.workflows[workflow_name]),
                        "status": "initiated"
                    }
                    await self.send_message(
                        recipients=message.sender,
                        content=response,
                        metadata={"in_response_to": message.id}
                    )
                    return
        
        # Default response
        await self.send_message(
            recipients=message.sender,
            content={
                "status": "acknowledged",
                "managed_agents": len(self.managed_agents),
                "workflows_available": list(self.workflows.keys())
            },
            metadata={"in_response_to": message.id}
        )
