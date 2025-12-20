"""
exceptions.py - Custom Exceptions for neuron_core

Hierarchy of custom exceptions for the framework.
"""


class NeuronException(Exception):
    """Base exception for all neuron_core errors."""
    pass


# Core System Exceptions

class ConfigurationError(NeuronException):
    """Raised when there's an error in configuration."""
    pass


class ValidationError(NeuronException):
    """Raised when validation fails."""
    pass


# Agent-Related Exceptions

class AgentError(NeuronException):
    """Base class for all agent-related exceptions."""
    pass


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize properly."""
    pass


class AgentProcessingError(AgentError):
    """Raised when an agent encounters an error during message processing."""
    pass


class AgentCapabilityError(AgentError):
    """Raised when an agent is asked to perform a capability it doesn't have."""
    pass


# Memory-Related Exceptions

class MemoryError(NeuronException):
    """Base class for all memory-related exceptions."""
    pass


class MemoryAccessError(MemoryError):
    """Raised when there's an error accessing memory."""
    pass


class MemoryStorageError(MemoryError):
    """Raised when there's an error storing information in memory."""
    pass


# Communication-Related Exceptions

class CommunicationError(NeuronException):
    """Base class for all communication-related exceptions."""
    pass


class SynapticBusError(CommunicationError):
    """Raised when there's an error in the SynapticBus communication system."""
    pass


class MessageValidationError(CommunicationError):
    """Raised when a message fails validation."""
    pass


class ChannelError(CommunicationError):
    """Raised when there's an error with a specific communication channel."""
    pass


# Circuit-Related Exceptions

class CircuitError(NeuronException):
    """Base class for all circuit-related exceptions."""
    pass


class CircuitDesignerError(CircuitError):
    """Raised when there's an error in circuit design or validation."""
    pass


class CircuitRuntimeError(CircuitError):
    """Raised when a circuit encounters a runtime error."""
    pass
