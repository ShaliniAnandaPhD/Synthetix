#!/usr/bin/env python3
"""
referee_bot.py - NFL Referee AI Bot

Analyzes video clips and cites specific NFL rules to determine penalties.
Uses Gemini 1.5 Flash for fast, cost-effective video analysis.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part


def analyze_play(
    video_path: str, 
    rule_context: str,
    project: str = "leafy-sanctuary-476515-t2",
    location: str = "us-central1"
) -> str:
    """
    Analyze a video clip and cite NFL rules for potential penalties.
    
    Args:
        video_path: Path to the video file
        rule_context: NFL rules text to reference
        project: GCP project ID
        location: GCP region
        
    Returns:
        Model's analysis as text
    """
    # Initialize Vertex AI
    vertexai.init(project=project, location=location)
    
    # Load the model - using gemini-1.5-flash for fast video analysis
    model = GenerativeModel("gemini-1.5-flash-001")
    
    # Read the video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Determine mime type based on extension
    ext = os.path.splitext(video_path)[1].lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska'
    }
    mime_type = mime_types.get(ext, 'video/mp4')
    
    # Load video as Part
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    video_part = Part.from_data(video_data, mime_type=mime_type)
    
    # Build the referee prompt
    prompt = f"""You are an NFL Referee. Analyze this video clip. 
Based ONLY on the following rules, determine if a penalty occurred and cite the rule section.

Rules:
{rule_context}

Provide your analysis in this format:
1. PLAY DESCRIPTION: Briefly describe what happened in the play
2. PENALTY DETERMINATION: Was there a penalty? (YES/NO)
3. RULE CITATION: If yes, cite the specific rule section
4. EXPLANATION: Explain why the penalty applies or why no penalty occurred
5. SIGNAL: What hand signal would the referee make?
"""
    
    # Generate response
    print("🏈 Analyzing play...")
    response = model.generate_content([video_part, prompt])
    
    return response.text


def load_rules(rules_path: str) -> str:
    """Load rules from a text file."""
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Main entry point for the Referee Bot."""
    parser = argparse.ArgumentParser(
        description="NFL Referee Bot - Analyze plays and cite rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/referee_bot.py --video clips/play1.mp4 --rules data/nfl_rules.txt
  python scripts/referee_bot.py --video touchdown.mp4 --rules rules.txt
        """
    )
    
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video clip to analyze"
    )
    parser.add_argument(
        "--rules",
        required=True,
        help="Path to the NFL rules text file"
    )
    parser.add_argument(
        "--project",
        default="leafy-sanctuary-476515-t2",
        help="GCP project ID"
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP region"
    )
    
    args = parser.parse_args()
    
    print("🎬 NFL Referee Bot")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Rules: {args.rules}")
    print("=" * 50)
    
    try:
        # Load the rules
        rule_context = load_rules(args.rules)
        print(f"📜 Loaded {len(rule_context)} characters of rules")
        
        # Analyze the play
        result = analyze_play(
            video_path=args.video,
            rule_context=rule_context,
            project=args.project,
            location=args.location
        )
        
        print("\n" + "=" * 50)
        print("🏁 REFEREE DECISION")
        print("=" * 50)
        print(result)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
