"""
validator_agent.py - Content Validation Agent for neuron_core

A safety-focused agent that validates content against compliance rules
before allowing it to be published or transmitted.
"""

import logging
from typing import Any, Dict, List

from .reflex_agent import ReflexAgent
from neuron_core.types import Message

logger = logging.getLogger(__name__)


# Banned concepts for compliance checking
BANNED_CONCEPTS: List[str] = [
    'fake',
    'simulated',
    'hallucinate',
    'error',
    'debug'
]


class ValidatorAgent(ReflexAgent):
    """
    A content validation agent for compliance and safety.
    
    Checks content against a list of banned concepts and filters
    unsafe content before it can be published or transmitted.
    
    Usage:
        validator = ValidatorAgent(name="ContentGuard")
        result = validator.validate("Some content to check")
        
        if result['is_safe']:
            publish(result['content'])
        else:
            log_violation(result['reason'])
    """
    
    def __init__(
        self,
        name: str = "ValidatorAgent",
        banned_concepts: List[str] = None,
        **kwargs
    ):
        """
        Initialize the Validator Agent.
        
        Args:
            name: Agent name
            banned_concepts: Optional custom list of banned concepts
            **kwargs: Additional args passed to ReflexAgent
        """
        super().__init__(name=name, **kwargs)
        
        # Use custom banned concepts or defaults
        self.banned_concepts = banned_concepts or BANNED_CONCEPTS
        
        # Add the validation rule
        self.add_rule("validate_content", self._validate_rule)
        
        logger.info(f"ValidatorAgent '{name}' initialized with {len(self.banned_concepts)} banned concepts")
    
    def _validate_rule(self, msg) -> Dict[str, Any]:
        """
        Rule handler for content validation.
        
        Processes all incoming messages through validation.
        """
        content = msg.content if hasattr(msg, 'content') else str(msg)
        return self.validate(content)
    
    def validate(self, content: str) -> Dict[str, Any]:
        """
        Validate content against banned concepts.
        
        Args:
            content: Text content to validate
            
        Returns:
            Dictionary with validation result:
            - If safe: {'is_safe': True, 'content': content}
            - If unsafe: {'is_safe': False, 'reason': '...', 'filtered_content': '[REDACTED]'}
        """
        if not content:
            return {'is_safe': True, 'content': content}
        
        # Check for banned concepts (case-insensitive)
        content_lower = content.lower()
        
        for concept in self.banned_concepts:
            if concept.lower() in content_lower:
                logger.warning(f"Compliance violation detected: '{concept}' found in content")
                return {
                    'is_safe': False,
                    'reason': f'Compliance Violation: {concept}',
                    'filtered_content': '[REDACTED]',
                    'original_length': len(content),
                    'violation': concept
                }
        
        # Content is safe
        return {
            'is_safe': True,
            'content': content
        }
    
    def validate_batch(self, contents: List[str]) -> List[Dict[str, Any]]:
        """
        Validate multiple content items.
        
        Args:
            contents: List of content strings to validate
            
        Returns:
            List of validation results
        """
        return [self.validate(content) for content in contents]
    
    def add_banned_concept(self, concept: str) -> None:
        """Add a new banned concept."""
        if concept.lower() not in [c.lower() for c in self.banned_concepts]:
            self.banned_concepts.append(concept)
            logger.info(f"Added banned concept: '{concept}'")
    
    def remove_banned_concept(self, concept: str) -> bool:
        """Remove a banned concept."""
        for i, c in enumerate(self.banned_concepts):
            if c.lower() == concept.lower():
                self.banned_concepts.pop(i)
                logger.info(f"Removed banned concept: '{concept}'")
                return True
        return False
    
    def get_banned_concepts(self) -> List[str]:
        """Get the current list of banned concepts."""
        return self.banned_concepts.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        base_info = super().get_agent_info() if hasattr(super(), 'get_agent_info') else {}
        base_info.update({
            "agent_type": "ValidatorAgent",
            "banned_concepts_count": len(self.banned_concepts),
            "capabilities": ["content_validation", "compliance_check"]
        })
        return base_info
