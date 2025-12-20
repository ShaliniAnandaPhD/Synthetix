"""
reflex_agent.py - ReflexAgent for neuron_core

A reactive agent that maps inputs directly to actions based on rules.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from neuron_core.types import Message, AgentState, AgentCapability, MemoryType
from neuron_core.memory.memory_manager import MemoryManager, InMemoryStore
from neuron_core.core.instrumentation import traced

logger = logging.getLogger(__name__)


class ReflexAgent:
    """
    A reactive agent that maps inputs directly to actions based on rules.
    
    ReflexAgent is the simplest type of neuron agent, ideal for:
    - Input classification
    - Simple routing
    - Immediate filtering
    - Formatting/Normalization
    """
    
    def __init__(
        self, 
        name: str,
        capabilities: Optional[List[AgentCapability]] = None,
        memory_store: Optional[Any] = None
    ):
        """
        Initialize a ReflexAgent.
        
        Args:
            name: Unique name for this agent
            capabilities: List of capabilities (optional)
            memory_store: Persistence backend (optional)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.state = AgentState.INITIALIZING
        self.message_count = 0
        
        # Initialize memory - use FirestoreMemoryStore directly for key-value storage
        self._memory_store = None
        if memory_store:
            self._memory_store = memory_store
        else:
            try:
                from neuron_core.memory.firestore_store import FirestoreMemoryStore
                self._memory_store = FirestoreMemoryStore(collection_name='agent_memory')
            except Exception as e:
                logger.warning(f"Could not initialize Firestore: {e}. Using in-memory fallback.")
                self._memory_store = _DictStore()

        self.capabilities = capabilities or []
        self.rules: Dict[str, Callable[[Message], Any]] = {}
        
        logger.info(f"ReflexAgent {name} initialized")

    def initialize(self) -> None:
        """Initialize agent resources and save initial state."""
        self.state = AgentState.READY
        self._save_state()
        logger.info(f"ReflexAgent {self.name} ready")

    def get_state(self) -> AgentState:
        """Return current agent state."""
        return self.state

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities."""
        return self.capabilities

    def add_rule(self, name: str, action: Callable[[Message], Any]) -> None:
        """
        Add a stimulus-response rule.
        
        Args:
            name: Identifier for the rule
            action: Function taking a Message and returning a result
        """
        self.rules[name] = action
        self._save_state()

    @traced(span_name="agent.process_rules")
    def process(self, message: Any) -> Dict[str, Any]:
        """
        Process an input message through all defined rules.
        
        Args:
            message: Input string or Message object
            
        Returns:
            Dictionary of results from triggered rules
        """
        self.state = AgentState.PROCESSING
        self.message_count += 1
        
        # Normalize input to Message object
        if isinstance(message, str):
            msg_obj = Message.create(
                sender="system",
                recipients=[self.id],
                content=message
            )
        else:
            msg_obj = message
            
        results = {}
        
        # Execute all rules (Reflex logic: check all conditions)
        for rule_name, action in self.rules.items():
            try:
                result = action(msg_obj)
                results[rule_name] = result
            except Exception as e:
                logger.error(f"Error executing rule {rule_name}: {e}")
                results[rule_name] = {"error": str(e)}
                
        self.state = AgentState.READY
        
        # Persist the interaction
        self._save_state()
        
        return results

    def _save_state(self) -> None:
        """Persist current agent state to memory."""
        state_data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "state": self.state.name,
            "rules": list(self.rules.keys()),
            "message_count": self.message_count,
            "last_updated": datetime.now().timestamp()
        }
        
        try:
            if hasattr(self._memory_store, 'store'):
                self._memory_store.store(self.name, state_data)
        except Exception as e:
            logger.warning(f"Failed to persist state for {self.name}: {e}")


class _DictStore:
    """Simple dict-based fallback store."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
    
    def store(self, key: str, value: Dict[str, Any]) -> bool:
        self._data[key] = value
        return True
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)
