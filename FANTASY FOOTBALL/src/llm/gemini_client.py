"""
Gemini client for Vertex AI.
Uses Google's Generative AI SDK with Vertex AI backend.
Imports are lazy to work with Modal's container lifecycle.
"""

import os
from typing import Optional, List, Dict


class GeminiAgent:
    """Client for interacting with Gemini on Vertex AI."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro-002"
    ):
        """
        Initialize the GeminiAgent.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location (default: us-central1)
            model_name: Model identifier for Gemini on Vertex AI
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Lazy import to work with Modal's container lifecycle
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Store for later use
        self._GenerativeModel = GenerativeModel
        
        # Create model instance
        self.model = GenerativeModel(model_name)
    
    def send_message(
        self,
        system_instruction: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 1.0
    ) -> str:
        """
        Send a message to the Gemini model and return the response.
        
        Args:
            system_instruction: System prompt defining the agent's role
            user_message: User's message/prompt
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            The model's text response.
        """
        try:
            # Create model with system instruction
            model_with_system = self._GenerativeModel(
                self.model_name,
                system_instruction=[system_instruction]
            )
            
            # Generate response
            response = model_with_system.generate_content(
                user_message,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            
            # Extract text from response
            if response.text:
                return response.text
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Error sending message to Gemini: {e}")
    
    def send_conversation(
        self,
        system_instruction: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 1.0
    ) -> str:
        """
        Send a multi-turn conversation to the Gemini model.
        
        Args:
            system_instruction: System prompt defining the agent's role
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            The model's text response.
        """
        try:
            # Create model with system instruction
            model_with_system = self._GenerativeModel(
                self.model_name,
                system_instruction=[system_instruction]
            )
            
            # Start a chat session
            chat = model_with_system.start_chat()
            
            # Send all messages except the last one as history
            for msg in messages[:-1]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    chat.send_message(content, stream=False)
                # Note: Gemini chat automatically handles assistant responses
            
            # Send the last message and get response
            if messages:
                last_msg = messages[-1]
                response = chat.send_message(
                    last_msg.get("content", ""),
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    stream=False
                )
                
                if response.text:
                    return response.text
                    
            return ""
                
        except Exception as e:
            raise RuntimeError(f"Error sending conversation to Gemini: {e}")
