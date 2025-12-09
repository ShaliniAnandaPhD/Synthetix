#!/usr/bin/env python3
"""
Main orchestrator for AI-powered fantasy football debate.
Demonstrates a 3-turn debate between city-specific agents.
"""

import os
import sys
import time
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.core.agent_factory import AgentFactory
from src.core.tempo_engine import TempoEngine
from src.core import lexical_injector
from src.llm.vertex_client import VertexAgent


class DebateOrchestrator:
    """Orchestrates multi-agent debates with city-specific personalities."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the debate orchestrator.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize core components
        self.factory = AgentFactory()
        self.tempo = TempoEngine()
        
        # Store agent configurations
        self.agents = {}
    
    def setup_agent(self, city_name: str, model_name: str = "claude-3-5-sonnet@20240620"):
        """
        Setup an agent for a specific city.
        
        Args:
            city_name: Name of the city (e.g., 'Philadelphia')
            model_name: Claude model identifier on Vertex AI
        """
        # Create Vertex AI client
        vertex_agent = VertexAgent(
            project_id=self.project_id,
            location=self.location,
            model_name=model_name
        )
        
        # Get system prompt from factory
        system_prompt = self.factory.construct_system_prompt(city_name)
        
        # Store agent configuration
        self.agents[city_name] = {
            'vertex_agent': vertex_agent,
            'system_prompt': system_prompt,
            'conversation_history': []
        }
        
        print(f"✓ Setup {city_name} agent")
    
    def get_response(
        self,
        city_name: str,
        user_message: str,
        show_thinking: bool = True
    ) -> str:
        """
        Get a response from a city agent with tempo and lexical flavor.
        
        Args:
            city_name: Name of the city agent
            user_message: Message to send to the agent
            show_thinking: Whether to show thinking delay
        
        Returns:
            The agent's response with lexical flavor applied.
        """
        if city_name not in self.agents:
            raise ValueError(f"Agent for {city_name} not setup. Call setup_agent() first.")
        
        agent_config = self.agents[city_name]
        
        # Get delay from tempo engine
        delay = self.tempo.get_delay(city_name)
        
        if show_thinking:
            print(f"  [{city_name} thinking for {delay:.2f}s...]", end='', flush=True)
        
        # Simulate thinking time
        time.sleep(delay)
        
        if show_thinking:
            print("\r" + " " * 60 + "\r", end='', flush=True)  # Clear thinking line
        
        # Get raw response from LLM
        raw_response = agent_config['vertex_agent'].send_message(
            system_instruction=agent_config['system_prompt'],
            user_message=user_message,
            max_tokens=512,
            temperature=1.0
        )
        
        # Apply lexical flavor
        flavored_response = lexical_injector.inject_flavor(raw_response, city_name)
        
        # Store in conversation history
        agent_config['conversation_history'].append({
            'user': user_message,
            'assistant': flavored_response
        })
        
        return flavored_response
    
    def run_debate(
        self,
        topic: str,
        agent1_city: str,
        agent2_city: str,
        num_turns: int = 3
    ):
        """
        Run a structured debate between two city agents.
        
        Args:
            topic: The debate topic
            agent1_city: First agent's city
            agent2_city: Second agent's city
            num_turns: Number of debate turns (default: 3)
        """
        print("\n" + "=" * 80)
        print(f"DEBATE: {topic}")
        print("=" * 80)
        print(f"\nParticipants:")
        print(f"  • {agent1_city} (Aggressive, Eye Test Focused)")
        print(f"  • {agent2_city} (Analytical, Data-Driven)")
        print("\n" + "-" * 80 + "\n")
        
        # Initial prompt for agent 1
        initial_prompt = f"Debate topic: {topic}\n\nGive your opening argument. Be concise (2-3 sentences)."
        
        print(f"🗣️  {agent1_city}:")
        agent1_response = self.get_response(agent1_city, initial_prompt)
        print(f"   {agent1_response}\n")
        
        # Debate loop
        for turn in range(num_turns):
            print(f"--- Turn {turn + 1} ---\n")
            
            # Agent 2 responds to Agent 1
            rebuttal_prompt = (
                f"Your opponent ({agent1_city}) just said:\n\"{agent1_response}\"\n\n"
                f"Provide your counter-argument. Be concise (2-3 sentences)."
            )
            
            print(f"🗣️  {agent2_city}:")
            agent2_response = self.get_response(agent2_city, rebuttal_prompt)
            print(f"   {agent2_response}\n")
            
            # Check if agent1 should interrupt (based on confidence)
            # For simplicity, we'll assume moderate confidence
            should_interrupt = self.tempo.check_interruption(agent1_city, opponent_confidence=0.6)
            
            if should_interrupt and turn < num_turns - 1:
                print(f"   [⚡ {agent1_city} interrupts!]\n")
            
            # Agent 1 responds back (if not last turn)
            if turn < num_turns - 1:
                counter_prompt = (
                    f"Your opponent ({agent2_city}) just said:\n\"{agent2_response}\"\n\n"
                    f"Provide your counter-argument. Be concise (2-3 sentences)."
                )
                
                print(f"🗣️  {agent1_city}:")
                agent1_response = self.get_response(agent1_city, counter_prompt)
                print(f"   {agent1_response}\n")
        
        print("-" * 80)
        print("🏁 Debate concluded!\n")


def main():
    """Main entry point."""
    
    # Get Google Cloud project ID from environment
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    
    if not project_id:
        print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.")
        print("\nPlease set it with:")
        print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("AI FANTASY FOOTBALL DEBATE SYSTEM")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = DebateOrchestrator(project_id=project_id)
    
    # Setup agents
    print("\n📋 Setting up agents...")
    orchestrator.setup_agent("Philadelphia")
    orchestrator.setup_agent("San Francisco")
    
    # Run debate
    orchestrator.run_debate(
        topic="Is Brock Purdy an elite quarterback?",
        agent1_city="Philadelphia",
        agent2_city="San Francisco",
        num_turns=3
    )
    
    print("\n✓ Debate complete!")


if __name__ == "__main__":
    main()
