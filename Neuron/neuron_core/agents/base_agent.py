"""
base_agent.py - Base Agent Class for neuron_core

Defines the abstract BaseAgent that all agent types inherit from.
"""

import asyncio
import functools
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from ..types import (
    AgentCapability,
    AgentID,
    AgentMetrics,
    AgentState,
    Message,
    MessageID,
    MessagePriority,
)
from ..exceptions import (
    AgentCapabilityError,
    AgentInitializationError,
    AgentProcessingError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def capability(name: str, description: str, 
               input_schema: Optional[Dict] = None,
               output_schema: Optional[Dict] = None):
    """
    Decorator to mark a method as an agent capability.
    
    Args:
        name: Name of the capability
        description: Human-readable description
        input_schema: Optional JSON schema for inputs
        output_schema: Optional JSON schema for outputs
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        
        wrapper._capability_info = AgentCapability(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema
        )
        return wrapper
    return decorator


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    agent_type: Type['BaseAgent']
    agent_id: Optional[AgentID] = None
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_params: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in neuron_core.
    
    This abstract class defines the core functionality and interfaces
    that all agents must implement. It handles message processing,
    capability registration, and lifecycle management.
    """
    
    def __init__(self, agent_id: Optional[AgentID] = None, name: str = "",
                 description: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for this agent (generated if not provided)
            name: Human-readable name for this agent
            description: Detailed description of this agent's purpose
            metadata: Additional contextual information
        """
        self.id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description
        self.metadata = metadata or {}
        
        # State management
        self._state = AgentState.INITIALIZING
        self._state_lock = threading.Lock()
        
        # Message handling
        self._message_queue: asyncio.Queue = None
        self._processing_task: Optional[asyncio.Task] = None
        
        # Capabilities
        self._capabilities: Dict[str, AgentCapability] = {}
        self._register_capabilities()
        
        # Metrics
        self._metrics = AgentMetrics()
        self._start_time: Optional[float] = None
        
        # Dependencies (set during initialization)
        self._synaptic_bus = None
        self._memory_manager = None
        self._initialized = False
        
        logger.debug(f"Agent {self.name} ({self.id}) created in INITIALIZING state")
    
    def _register_capabilities(self) -> None:
        """Discover and register capabilities from methods."""
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
                if callable(attr) and hasattr(attr, '_capability_info'):
                    cap_info = attr._capability_info
                    self._capabilities[cap_info.name] = cap_info
            except Exception:
                pass
    
    def initialize(self, synaptic_bus: Any = None, memory_manager: Any = None) -> None:
        """
        Initialize the agent with dependencies.
        
        Args:
            synaptic_bus: Communication system for messaging
            memory_manager: Memory system for storage
        """
        if self._initialized:
            logger.warning(f"Agent {self.id} is already initialized")
            return
        
        self._synaptic_bus = synaptic_bus
        self._memory_manager = memory_manager
        
        # Create async queue for messages
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        self._message_queue = asyncio.Queue()
        
        # Agent-specific initialization
        self._initialize()
        
        with self._state_lock:
            self._state = AgentState.READY
        
        self._initialized = True
        logger.info(f"Agent {self.name} ({self.id}) initialized and ready")
    
    def _initialize(self) -> None:
        """Agent-specific initialization. Override in subclasses."""
        pass
    
    def start(self) -> None:
        """Start the agent's message processing."""
        with self._state_lock:
            if self._state != AgentState.READY:
                raise AgentInitializationError(
                    f"Agent {self.id} cannot start from state {self._state.name}"
                )
            self._state = AgentState.PROCESSING
        
        self._start_time = time.time()
        logger.info(f"Agent {self.name} ({self.id}) started processing")
    
    def stop(self) -> None:
        """Stop the agent's message processing."""
        with self._state_lock:
            if self._state == AgentState.PROCESSING:
                self._state = AgentState.PAUSED
        
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None
        
        logger.info(f"Agent {self.name} ({self.id}) stopped")
    
    def terminate(self) -> None:
        """Permanently terminate the agent."""
        self.stop()
        
        with self._state_lock:
            self._state = AgentState.TERMINATED
        
        self._cleanup()
        logger.info(f"Agent {self.name} ({self.id}) terminated")
    
    def _cleanup(self) -> None:
        """Cleanup when agent is terminated. Override in subclasses."""
        pass
    
    async def receive_message(self, message: Message) -> None:
        """
        Receive a message for processing.
        
        Args:
            message: The message to process
        """
        if self._message_queue:
            await self._message_queue.put(message)
    
    async def send_message(self, recipients: Union[AgentID, List[AgentID]], 
                          content: Any,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          metadata: Optional[Dict[str, Any]] = None,
                          trace_id: Optional[str] = None) -> None:
        """
        Send a message to other agents.
        
        Args:
            recipients: ID or list of IDs of recipient agents
            content: Message content/payload
            priority: Message priority level
            metadata: Additional message metadata
            trace_id: Optional trace ID for message tracking
        """
        if isinstance(recipients, str):
            recipients = [recipients]
        
        message = Message.create(
            sender=self.id,
            recipients=recipients,
            content=content,
            priority=priority,
            metadata=metadata,
            trace_id=trace_id
        )
        
        if self._synaptic_bus:
            await self._synaptic_bus.send(message)
        else:
            logger.warning(f"Agent {self.id} has no synaptic bus, message not sent")
    
    @abstractmethod
    async def process_message(self, message: Message) -> None:
        """
        Process a received message. Must be implemented by subclasses.
        
        Args:
            message: The message to process
        """
        pass
    
    def get_state(self) -> AgentState:
        """Get the current agent state."""
        with self._state_lock:
            return self._state
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        if self._start_time:
            self._metrics.uptime = time.time() - self._start_time
        return self._metrics
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities."""
        return list(self._capabilities.values())
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        return capability_name in self._capabilities
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id}, name={self.name}, state={self._state.name})>"


class AgentBuilder:
    """Builder pattern for creating agents."""
    
    def __init__(self, agent_manager: Any = None):
        self._agent_manager = agent_manager
        self._config = AgentConfig(agent_type=BaseAgent)
    
    def of_type(self, agent_type: Type[BaseAgent]) -> 'AgentBuilder':
        self._config.agent_type = agent_type
        return self
    
    def with_id(self, agent_id: AgentID) -> 'AgentBuilder':
        self._config.agent_id = agent_id
        return self
    
    def with_name(self, name: str) -> 'AgentBuilder':
        self._config.name = name
        return self
    
    def with_description(self, description: str) -> 'AgentBuilder':
        self._config.description = description
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'AgentBuilder':
        self._config.metadata = metadata
        return self
    
    def with_config(self, **kwargs) -> 'AgentBuilder':
        self._config.config_params.update(kwargs)
        return self
    
    def build(self) -> AgentID:
        """Build and register the agent."""
        if self._agent_manager:
            return self._agent_manager.create_agent(self._config)
        
        # Create agent directly if no manager
        agent = self._config.agent_type(
            agent_id=self._config.agent_id,
            name=self._config.name,
            description=self._config.description,
            metadata=self._config.metadata
        )
        agent.initialize()
        return agent.id
