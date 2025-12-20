"""
neuron_core - A Composable Agent Framework Library

This is the core engine of the Neuron framework, extracted for deployment
to Google Vertex AI Agent Engine.

Provides:
- Agent base classes (ReflexAgent, DeliberativeAgent, LearningAgent, CoordinatorAgent)
- Memory management system
- Inter-agent communication via SynapticBus
"""

__version__ = "1.0.0"
__author__ = "Neuron Framework Team"

# Core types
from .types import (
    AgentID,
    MessageID,
    CircuitID,
    MemoryID,
    Message,
    MessagePriority,
    AgentState,
    AgentCapability,
    AgentMetrics,
    MemoryType,
    MemoryEntry,
)

# Exceptions
from .exceptions import (
    NeuronException,
    AgentError,
    AgentInitializationError,
    AgentProcessingError,
    AgentCapabilityError,
    MemoryAccessError,
    MemoryStorageError,
    CommunicationError,
    ValidationError,
)

# Agent classes
from .agents import (
    BaseAgent,
    ReflexAgent,
    DeliberativeAgent,
    LearningAgent,
    CoordinatorAgent,
)

# Core systems
from .core import SynapticBus, Channel, MessageRouter
from .memory import MemoryManager, MemoryStore, InMemoryStore

__all__ = [
    # Types
    "AgentID",
    "MessageID", 
    "CircuitID",
    "MemoryID",
    "Message",
    "MessagePriority",
    "AgentState",
    "AgentCapability",
    "AgentMetrics",
    "MemoryType",
    "MemoryEntry",
    # Exceptions
    "NeuronException",
    "AgentError",
    "AgentInitializationError",
    "AgentProcessingError",
    "AgentCapabilityError",
    "MemoryAccessError",
    "MemoryStorageError",
    "CommunicationError",
    "ValidationError",
    # Agents
    "BaseAgent",
    "ReflexAgent",
    "DeliberativeAgent",
    "LearningAgent",
    "CoordinatorAgent",
    # Core
    "SynapticBus",
    "Channel",
    "MessageRouter",
    # Memory
    "MemoryManager",
    "MemoryStore",
    "InMemoryStore",
]
