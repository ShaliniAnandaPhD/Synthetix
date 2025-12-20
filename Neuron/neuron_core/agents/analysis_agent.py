"""
analysis_agent.py - Video Analysis Agent for neuron_core

A specialized ReflexAgent that can analyze video content using
Gemini multimodal capabilities and cite specific rules.
"""

import os
import logging
from typing import Any, Dict, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from .reflex_agent import ReflexAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(ReflexAgent):
    """
    A specialized agent for video content analysis.
    
    Extends ReflexAgent with multimodal capabilities using Gemini 1.5 Flash
    for analyzing video clips and citing specific rules.
    
    Usage:
        agent = AnalysisAgent(name="RefereeBot")
        result = agent.analyze_content("play.mp4", "NFL Rules text...")
        
        # Or via message processing:
        agent.process("ANALYZE_VIDEO:play.mp4|data/nfl_rules.txt")
    """
    
    def __init__(
        self,
        name: str = "AnalysisAgent",
        project: str = "leafy-sanctuary-476515-t2",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-exp",
        **kwargs
    ):
        """
        Initialize the Analysis Agent.
        
        Args:
            name: Agent name
            project: GCP project ID
            location: GCP region
            model_name: Gemini model for analysis
            **kwargs: Additional args passed to ReflexAgent
        """
        super().__init__(name=name, **kwargs)
        
        self.project = project
        self.location = location
        self.model_name = model_name
        self._model = None
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=project, location=location)
            self._model = GenerativeModel(model_name)
            logger.info(f"AnalysisAgent initialized with {model_name}")
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
        
        # Add the video analysis rule
        self.add_rule("analyze_video", self._analyze_video_rule)
        
        logger.info(f"AnalysisAgent '{name}' ready")
    
    def _analyze_video_rule(self, msg) -> Dict[str, Any]:
        """
        Rule handler for video analysis requests.
        
        Message format: ANALYZE_VIDEO:video_path|rules_path
        """
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        if not content.startswith("ANALYZE_VIDEO:"):
            return {"skipped": True, "reason": "Not an analysis request"}
        
        try:
            # Parse the command: ANALYZE_VIDEO:video.mp4|rules.txt
            payload = content.replace("ANALYZE_VIDEO:", "")
            
            if "|" in payload:
                video_path, rules_path = payload.split("|", 1)
                
                # Load rules from file
                if os.path.exists(rules_path):
                    with open(rules_path, 'r') as f:
                        context = f.read()
                else:
                    context = rules_path  # Treat as inline rules
            else:
                video_path = payload
                context = "Standard NFL rules apply."
            
            # Run analysis
            result = self.analyze_content(video_path.strip(), context.strip())
            
            return {
                "analysis": result,
                "video": video_path,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def analyze_content(self, video_path: str, context: str) -> str:
        """
        Analyze video content and cite specific rules.
        
        Args:
            video_path: Path to the video file
            context: Rules/context to reference for analysis
            
        Returns:
            Analysis result as text
        """
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Determine MIME type
        ext = os.path.splitext(video_path)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska'
        }
        mime_type = mime_types.get(ext, 'video/mp4')
        
        # Load video
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_part = Part.from_data(video_data, mime_type=mime_type)
        
        # Build referee prompt
        prompt = f"""You are an NFL Referee. Analyze this video clip.
Based ONLY on the following rules, determine if a penalty occurred and cite the rule section.

Rules:
{context}

Provide your analysis in this format:
1. PLAY DESCRIPTION: Briefly describe what happened in the play
2. PENALTY DETERMINATION: Was there a penalty? (YES/NO)
3. RULE CITATION: If yes, cite the specific rule section
4. EXPLANATION: Explain why the penalty applies or why no penalty occurred
5. SIGNAL: What hand signal would the referee make?
"""
        
        logger.info(f"Analyzing video: {video_path}")
        
        # Generate response
        response = self._model.generate_content([video_part, prompt])
        
        logger.info("Analysis complete")
        return response.text
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information including model details."""
        base_info = super().get_agent_info() if hasattr(super(), 'get_agent_info') else {}
        base_info.update({
            "agent_type": "AnalysisAgent",
            "model": self.model_name,
            "capabilities": ["video_analysis", "rule_citation"]
        })
        return base_info
