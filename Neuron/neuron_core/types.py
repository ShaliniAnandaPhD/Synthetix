"""
types.py - Core Type Definitions for neuron_core

Fundamental type definitions, enums, and data classes used throughout the framework.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TypeVar

# Type variables for generic programming
T = TypeVar('T')

# Core identifiers
AgentID = str          # Unique identifier for an agent
MessageID = str        # Unique identifier for a message
CircuitID = str        # Unique identifier for a circuit of agents
SignalID = str         # Unique identifier for a signal
MemoryID = str         # Unique identifier for a memory entry


class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    URGENT = auto()


class AgentState(Enum):
    """Possible states of an Agent."""
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    PAUSED = auto()
    TERMINATED = auto()


class MemoryType(Enum):
    """Types of memory in the framework."""
    WORKING = auto()
    EPISODIC = auto()
    SEMANTIC = auto()
    PROCEDURAL = auto()
    EMOTIONAL = auto()


class CircuitRole(Enum):
    """Roles an agent can play within a circuit."""
    INPUT = auto()
    PROCESSOR = auto()
    COORDINATOR = auto()
    MEMORY = auto()
    OUTPUT = auto()
    MONITOR = auto()


@dataclass
class Message:
    """Basic unit of communication between agents."""
    id: MessageID
    sender: AgentID
    recipients: List[AgentID]
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    trace_id: Optional[str] = None
    
    @classmethod
    def create(cls, sender: AgentID, recipients: List[AgentID], content: Any, 
               priority: MessagePriority = MessagePriority.NORMAL,
               metadata: Optional[Dict[str, Any]] = None,
               trace_id: Optional[str] = None) -> 'Message':
        """Factory method to create a new message with a unique ID."""
        return cls(
            id=str(uuid.uuid4()),
            sender=sender,
            recipients=recipients,
            content=content,
            priority=priority,
            metadata=metadata or {},
            trace_id=trace_id or str(uuid.uuid4())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        result = asdict(self)
        result['priority'] = self.priority.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary."""
        data = data.copy()
        if 'priority' in data and isinstance(data['priority'], str):
            data['priority'] = MessagePriority[data['priority']]
        return cls(**data)


@dataclass
class AgentCapability:
    """Defines a specific capability that an agent possesses."""
    name: str
    description: str
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        return cls(**data)


@dataclass
class AgentMetrics:
    """Stores metrics about an agent's performance."""
    message_count: int = 0
    processing_time: float = 0
    error_count: int = 0
    memory_usage: float = 0
    cpu_usage: float = 0
    last_active: Optional[float] = None
    uptime: float = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_processing_time(self, time_delta: float) -> None:
        self.processing_time += time_delta
        self.last_active = time.time()
    
    def increment_message_count(self) -> None:
        self.message_count += 1
        self.last_active = time.time()
    
    def increment_error_count(self) -> None:
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    """Represents an entry in an agent's memory."""
    id: MemoryID
    content: Any
    memory_type: MemoryType
    created_at: float
    last_accessed: float
    access_count: int = 0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, content: Any, memory_type: MemoryType, 
               importance: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> 'MemoryEntry':
        now = time.time()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            created_at=now,
            last_accessed=now,
            importance=importance,
            metadata=metadata or {}
        )
    
    def access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['memory_type'] = self.memory_type.name
        return result


@dataclass
class CircuitDefinition:
    """Defines a circuit of connected agents."""
    id: CircuitID
    name: str
    description: str
    agents: Dict[AgentID, Dict[str, Any]]
    connections: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, name: str, description: str, 
               agents: Dict[AgentID, Dict[str, Any]],
               connections: List[Dict[str, Any]],
               metadata: Optional[Dict[str, Any]] = None) -> 'CircuitDefinition':
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agents=agents,
            connections=connections,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
