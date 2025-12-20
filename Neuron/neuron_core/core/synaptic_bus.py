"""
synaptic_bus.py - Communication System for neuron_core

Inter-agent messaging inspired by neural synapses.
"""

import asyncio
import logging
import queue
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..types import AgentID, Message, MessagePriority
from ..exceptions import ChannelError, CommunicationError, SynapticBusError

logger = logging.getLogger(__name__)


class DeliveryGuarantee(Enum):
    """Types of delivery guarantees for messages."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class Channel:
    """
    A named communication channel for groups of agents.
    
    Channels provide pub/sub communication between agents.
    """
    
    def __init__(self, name: str, buffer_size: int = 1000):
        self.name = name
        self._buffer_size = buffer_size
        self._subscribers: Set[AgentID] = set()
        self._message_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._lock = threading.Lock()
        self._active = True
    
    def subscribe(self, agent_id: AgentID) -> None:
        """Subscribe an agent to this channel."""
        if not self._active:
            raise ChannelError(f"Channel {self.name} is inactive")
        
        with self._lock:
            if agent_id in self._subscribers:
                raise ChannelError(f"Agent {agent_id} already subscribed")
            self._subscribers.add(agent_id)
        
        logger.debug(f"Agent {agent_id} subscribed to channel {self.name}")
    
    def unsubscribe(self, agent_id: AgentID) -> None:
        """Unsubscribe an agent from this channel."""
        with self._lock:
            if agent_id in self._subscribers:
                self._subscribers.remove(agent_id)
    
    def publish(self, message: Message) -> None:
        """Publish a message to this channel."""
        if not self._active:
            raise ChannelError(f"Channel {self.name} is inactive")
        
        try:
            self._message_queue.put_nowait(message)
        except queue.Full:
            raise ChannelError(f"Channel {self.name} queue is full")
    
    def get_message(self, timeout: float = 0.1) -> Optional[Message]:
        """Get the next message from this channel."""
        if not self._active:
            return None
        
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_subscribers(self) -> Set[AgentID]:
        """Get the set of subscribers."""
        with self._lock:
            return self._subscribers.copy()
    
    def message_count(self) -> int:
        """Get the number of pending messages."""
        return self._message_queue.qsize()
    
    def close(self) -> None:
        """Close this channel."""
        self._active = False


class MessageRouter:
    """Routes messages to their destinations."""
    
    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._routing_rules: List[tuple] = []
        self._lock = threading.Lock()
    
    def create_channel(self, name: str, buffer_size: int = 1000) -> Channel:
        """Create a new communication channel."""
        with self._lock:
            if name in self._channels:
                raise ChannelError(f"Channel {name} already exists")
            channel = Channel(name, buffer_size)
            self._channels[name] = channel
            return channel
    
    def remove_channel(self, name: str) -> None:
        """Remove a communication channel."""
        with self._lock:
            if name not in self._channels:
                raise ChannelError(f"Channel {name} does not exist")
            self._channels[name].close()
            del self._channels[name]
    
    def get_channel(self, name: str) -> Channel:
        """Get a channel by name."""
        with self._lock:
            if name not in self._channels:
                raise ChannelError(f"Channel {name} does not exist")
            return self._channels[name]
    
    def get_all_channels(self) -> Dict[str, Channel]:
        """Get all channels."""
        with self._lock:
            return self._channels.copy()
    
    def add_routing_rule(self, condition: Callable[[Message], bool],
                        targets: Union[AgentID, List[AgentID], str]) -> int:
        """Add a routing rule."""
        with self._lock:
            rule_id = len(self._routing_rules)
            self._routing_rules.append((condition, targets))
            return rule_id
    
    def route_message(self, message: Message) -> List[AgentID]:
        """Determine recipients for a message."""
        recipients = list(message.recipients) if message.recipients else []
        
        # Apply routing rules
        for condition, targets in self._routing_rules:
            try:
                if condition(message):
                    if isinstance(targets, str):
                        # It's a channel name
                        channel = self._channels.get(targets)
                        if channel:
                            recipients.extend(channel.get_subscribers())
                    elif isinstance(targets, list):
                        recipients.extend(targets)
                    else:
                        recipients.append(targets)
            except Exception as e:
                logger.error(f"Error applying routing rule: {e}")
        
        return list(set(recipients))  # Deduplicate


class SynapticBus:
    """
    Central communication system for inter-agent messaging.
    
    Provides message routing, channel management, and delivery.
    """
    
    def __init__(self, delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE):
        self._delivery_guarantee = delivery_guarantee
        self._router = MessageRouter()
        self._agents: Dict[AgentID, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._message_handlers: List[Callable[[Message], None]] = []
    
    def register_agent(self, agent_id: AgentID, agent: Any) -> None:
        """Register an agent with the bus."""
        with self._lock:
            self._agents[agent_id] = agent
        logger.debug(f"Agent {agent_id} registered with SynapticBus")
    
    def unregister_agent(self, agent_id: AgentID) -> None:
        """Unregister an agent from the bus."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
    
    def get_agent(self, agent_id: AgentID) -> Optional[Any]:
        """Get a registered agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)
    
    async def send(self, message: Message) -> None:
        """
        Send a message through the bus.
        
        Args:
            message: Message to send
        """
        recipients = self._router.route_message(message)
        
        for handler in self._message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        
        for recipient_id in recipients:
            agent = self.get_agent(recipient_id)
            if agent:
                try:
                    await agent.receive_message(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {recipient_id}: {e}")
            else:
                logger.warning(f"Agent {recipient_id} not found")
    
    def send_sync(self, message: Message) -> None:
        """Synchronous message sending."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send(message))
        except RuntimeError:
            asyncio.run(self.send(message))
    
    def create_channel(self, name: str, buffer_size: int = 1000) -> Channel:
        """Create a new communication channel."""
        return self._router.create_channel(name, buffer_size)
    
    def get_channel(self, name: str) -> Channel:
        """Get a channel by name."""
        return self._router.get_channel(name)
    
    def subscribe_to_channel(self, agent_id: AgentID, channel_name: str) -> None:
        """Subscribe an agent to a channel."""
        channel = self._router.get_channel(channel_name)
        channel.subscribe(agent_id)
    
    def add_message_handler(self, handler: Callable[[Message], None]) -> None:
        """Add a global message handler for monitoring."""
        self._message_handlers.append(handler)
    
    def start(self) -> None:
        """Start the SynapticBus."""
        self._running = True
        logger.info("SynapticBus started")
    
    def stop(self) -> None:
        """Stop the SynapticBus."""
        self._running = False
        logger.info("SynapticBus stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if the bus is running."""
        return self._running
