"""
code_agent.py - Code Execution Agent for neuron_core

An agent that can write and execute Python code in a secure sandbox
using Vertex AI's Code Execution capability.

The agent doesn't guess - it programs solutions and runs them.
"""

import logging
from typing import Dict, Any, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding

from .reflex_agent import ReflexAgent

logger = logging.getLogger(__name__)


class CodeAgent(ReflexAgent):
    """
    An agent capable of writing and executing Python code.
    
    Uses Vertex AI's Code Execution sandbox to run computations,
    data analysis, and complex calculations without hallucinating.
    
    Usage:
        agent = CodeAgent(name="Coder")
        result = agent.solve_problem("Calculate the 100th Fibonacci number")
        
        # Or via message processing
        result = agent.process("CODE:What is 2^1000?")
    """
    
    def __init__(
        self,
        name: str = "CodeAgent",
        project: str = "leafy-sanctuary-476515-t2",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-exp",
        **kwargs
    ):
        """
        Initialize the Code Execution Agent.
        
        Args:
            name: Agent name
            project: GCP project ID
            location: GCP region
            model_name: Gemini model for code execution
            **kwargs: Additional args passed to ReflexAgent
        """
        super().__init__(name=name, **kwargs)
        
        self.project = project
        self.location = location
        self.model_name = model_name
        self._model = None
        
        # Initialize Vertex AI and the model with code execution
        try:
            vertexai.init(project=project, location=location)
            
            # Configure model with code execution capability
            self._model = GenerativeModel(
                model_name,
                tools=[Tool.from_google_search_retrieval(
                    google_search_retrieval=grounding.GoogleSearchRetrieval()
                )]
            )
            
            # Try to use code execution if available
            try:
                from vertexai.generative_models import CodeExecution
                self._model = GenerativeModel(
                    model_name,
                    tools=[Tool(code_execution=CodeExecution())]
                )
                logger.info(f"CodeAgent initialized with code execution: {model_name}")
            except ImportError:
                # Fallback to model without explicit code execution tool
                logger.warning("CodeExecution not available, using standard model")
                self._model = GenerativeModel(model_name)
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self._model = None
        
        # Add the CODE: rule
        self.add_rule("code_solve", self._code_solve_rule)
        
        logger.info(f"CodeAgent '{name}' ready")
    
    def _code_solve_rule(self, msg) -> Dict[str, Any]:
        """
        Rule handler for CODE: messages.
        
        Format: CODE:problem_description
        """
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        if not content.upper().startswith("CODE:"):
            return {"skipped": True}
        
        problem = content[5:].strip()
        
        try:
            result = self.solve_problem(problem)
            return {
                "problem": problem,
                "solution": result,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Code solve failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def solve_problem(self, problem: str) -> str:
        """
        Solve a problem using code execution.
        
        The model will automatically write and execute Python code
        if needed to solve the problem.
        
        Args:
            problem: Problem description
            
        Returns:
            Solution with code output
        """
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        # Build prompt that encourages code execution
        prompt = f"""You are a computational problem solver. 
        
When given a problem, you should:
1. If it requires calculation or data processing, write Python code to solve it
2. Execute the code and provide the result
3. Explain your solution

Problem: {problem}

Please solve this problem. If it requires computation, write and run Python code to get the exact answer.
"""
        
        logger.info(f"🧮 Solving problem: {problem[:50]}...")
        
        try:
            response = self._model.generate_content(prompt)
            
            # Extract the text response
            result = response.text
            
            logger.info(f"✅ Solution generated ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Problem solving failed: {e}")
            raise
    
    def execute_code(self, code: str) -> str:
        """
        Execute Python code directly.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output
        """
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        prompt = f"""Execute the following Python code and return the output:

```python
{code}
```

Run this code and show me the result."""
        
        response = self._model.generate_content(prompt)
        return response.text
    
    def analyze_data(self, data_description: str, analysis_task: str) -> str:
        """
        Analyze data using code execution.
        
        Args:
            data_description: Description of the data
            analysis_task: What analysis to perform
            
        Returns:
            Analysis results
        """
        prompt = f"""You have the following data:
{data_description}

Task: {analysis_task}

Write Python code to perform this analysis and provide the results."""
        
        response = self._model.generate_content(prompt)
        return response.text
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        base_info = super().get_agent_info() if hasattr(super(), 'get_agent_info') else {}
        base_info.update({
            "agent_type": "CodeAgent",
            "model": self.model_name,
            "capabilities": ["code_execution", "computation", "data_analysis"]
        })
        return base_info
