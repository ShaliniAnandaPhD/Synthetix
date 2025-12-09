"""
Vertex AI client for Claude 3.5 Sonnet via Anthropic Vertex SDK.
"""

from anthropic import AnthropicVertex
from typing import Optional


class VertexAgent:
    """Client for interacting with Claude 3.5 Sonnet on Vertex AI."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "claude-3-5-sonnet@20240620"
    ):
        """
        Initialize the VertexAgent.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location (default: us-central1)
            model_name: Model identifier for Claude on Vertex AI
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize AnthropicVertex client
        self.client = AnthropicVertex(
            region=location,
            project_id=project_id
        )
    
    def send_message(
        self,
        system_instruction: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 1.0
    ) -> str:
        """
        Send a message to the Claude model and return the response.
        
        Args:
            system_instruction: System prompt defining the agent's role
            user_message: User's message/prompt
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            The model's text response.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_instruction,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Error sending message to Vertex AI: {e}")
    
    def send_conversation(
        self,
        system_instruction: str,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 1.0
    ) -> str:
        """
        Send a multi-turn conversation to the Claude model.
        
        Args:
            system_instruction: System prompt defining the agent's role
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            The model's text response.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_instruction,
                messages=messages
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Error sending conversation to Vertex AI: {e}")
