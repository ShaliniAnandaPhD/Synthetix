"""
core - Core infrastructure for neuron_core

Provides SynapticBus for inter-agent communication.
"""

from .synaptic_bus import SynapticBus, Channel, MessageRouter

__all__ = ["SynapticBus", "Channel", "MessageRouter"]
