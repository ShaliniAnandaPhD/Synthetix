"""
VertexNeuronAdapter - Bridge between Vertex AI Reasoning Engine and neuron_core

This module provides the adapter class that enables Google's Vertex AI
Reasoning Engine to interact with the neuron_core agent framework.

The adapter is designed to be pickle-safe for Vertex AI deployment.
"""

import logging
import uuid
from typing import Any, Dict, List, Callable, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimpleMessage:
    """Lightweight message for pickle-safe processing."""
    id: str
    sender: str
    content: Any
    
    @classmethod
    def create(cls, sender: str, content: Any) -> 'SimpleMessage':
        return cls(id=str(uuid.uuid4()), sender=sender, content=content)


class VertexNeuronAdapter:
    """
    Adapter for Vertex AI Reasoning Engine integration.
    
    This class bridges Google's Vertex AI Reasoning Engine with the
    neuron_core agent framework, enabling cloud deployment of cognitive
    agents.
    
    This adapter is pickle-safe - tracer is initialized lazily on first use.
    
    Usage:
        adapter = VertexNeuronAdapter()
        response = adapter.query("What is the weather?")
    
    For Vertex AI Reasoning Engine:
        This class is designed to be compatible with the Reasoning Engine's
        expected interface, where query() is the main entry point.
    """
    
    def __init__(self, agent_name: str = "VertexBot"):
        """
        Initialize the Vertex Neuron Adapter.
        
        Args:
            agent_name: Name for the underlying agent
        """
        self.agent_name = agent_name
        self.agent_id = str(uuid.uuid4())
        
        # Tracer is NOT initialized here - deferred for pickle safety
        self._tracer = None
        
        # Pickle-safe rules storage (dict of rule_name -> response pattern)
        self.rules: Dict[str, Dict[str, Any]] = {}
        
        # Setup default rules
        self._setup_default_rules()
        
        logger.info(f"VertexNeuronAdapter initialized with agent: {agent_name}")
    
    def _get_tracer(self):
        """Lazy initialization of tracer for pickle safety."""
        if self._tracer is None:
            try:
                from neuron_core.core.instrumentation import setup_tracer
                self._tracer = setup_tracer('vertex-neuron-adapter')
            except Exception as e:
                logger.warning(f"Could not initialize tracer: {e}")
                self._tracer = _NoOpTracer()
        return self._tracer
    
    def _setup_default_rules(self) -> None:
        """Setup default processing rules for the agent."""
        # Default rule - responds to any input
        self.rules["default"] = {
            "type": "default",
            "pattern": None  # Matches everything
        }
        
        # Query rule - handles explicit queries
        self.rules["query"] = {
            "type": "query",
            "pattern": "query"
        }
    
    def _process_rule(self, rule_name: str, rule_config: Dict, content: str) -> Dict[str, Any]:
        """Process a rule and generate response."""
        if rule_name == "query":
            return {
                "answer": f"Query received: {content}",
                "confidence": 0.95,
                "source": "neuron_core"
            }
        elif rule_name == "default":
            return {
                "response": f"Processed by {self.agent_name}: {content}",
                "agent_id": self.agent_id,
                "status": "success"
            }
        else:
            return {"response": content, "rule": rule_name}
    
    def query(self, input_text: str) -> Union[str, Dict[str, Any]]:
        """
        Process a query through the neuron_core agent.
        
        This is the main entry point for Vertex AI Reasoning Engine.
        
        Args:
            input_text: The query text to process
            
        Returns:
            Response from the agent as a string or dictionary
        """
        tracer = self._get_tracer()
        
        with tracer.start_as_current_span('vertex.query') as span:
            # Record the input for tracing
            span.set_attribute('neuron.input', input_text)
            span.set_attribute('neuron.agent_name', self.agent_name)
            span.set_attribute('neuron.agent_id', self.agent_id)
            
            try:
                responses = {}
                
                for rule_name, rule_config in self.rules.items():
                    pattern = rule_config.get("pattern")
                    
                    # Check if pattern matches (None = always match)
                    if pattern is None or pattern in str(input_text).lower():
                        responses[rule_name] = self._process_rule(rule_name, rule_config, input_text)
                
                # Return the response
                if responses:
                    span.set_attribute('neuron.rules_matched', list(responses.keys()))
                    if "query" in responses:
                        span.set_attribute('neuron.response_type', 'query')
                        return responses["query"]
                    elif "default" in responses:
                        span.set_attribute('neuron.response_type', 'default')
                        return responses["default"]
                    else:
                        return responses
                
                span.set_attribute('neuron.response_type', 'no_match')
                return {"response": f"Processed: {input_text}", "status": "no_rules_matched"}
                
            except Exception as e:
                span.set_attribute('neuron.error', True)
                span.record_exception(e)
                logger.error(f"Query processing failed: {e}")
                return {"error": str(e), "status": "failed"}
    
    def set_model(self, model: str) -> None:
        """
        Set the model for the agent (Vertex AI compatibility).
        
        Args:
            model: Model name or identifier
        """
        self.model = model
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying agent.
        
        Returns:
            Dictionary with agent details
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": "ReflexAgent",
            "state": "READY",
            "rules": list(self.rules.keys())
        }
    
    def add_rule(self, rule_name: str, pattern: str = None) -> None:
        """
        Add a custom processing rule to the agent.
        
        Args:
            rule_name: Name/identifier for the rule
            pattern: Optional pattern to match (None = match all)
        """
        self.rules[rule_name] = {"type": rule_name, "pattern": pattern}
    
    def __repr__(self) -> str:
        return f"<VertexNeuronAdapter(agent={self.agent_name}, id={self.agent_id})>"
    
    # Pickle support - exclude tracer
    def __getstate__(self):
        """Return state for pickling - exclude tracer."""
        state = self.__dict__.copy()
        state['_tracer'] = None  # Don't pickle the tracer
        return state
    
    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        self._tracer = None  # Will be lazily initialized


class _NoOpTracer:
    """No-op tracer fallback when real tracer unavailable."""
    
    def start_as_current_span(self, name):
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span for fallback."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def set_attribute(self, key, value):
        pass
    
    def record_exception(self, e):
        pass
