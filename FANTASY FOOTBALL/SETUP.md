# AI Fantasy Football Debate System - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `anthropic[vertex]` - Anthropic Claude SDK for Vertex AI
- `google-cloud-aiplatform` - Google Cloud AI Platform SDK

### 2. Configure Google Cloud

Set your Google Cloud project ID:

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
```

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

### 4. Run the Debate

```bash
python main.py
```

---

## 📁 Project Structure

```
FANTASY FOOTBALL/
├── config/
│   └── city_profiles.json          # City agent configurations
├── src/
│   ├── core/
│   │   ├── agent_factory.py        # Dynamic prompt construction
│   │   ├── tempo_engine.py         # Conversational timing
│   │   └── lexical_injector.py     # Phrase injection
│   └── llm/
│       └── vertex_client.py        # Vertex AI client
├── main.py                         # Debate orchestrator
├── requirements.txt                # Dependencies
└── test_core_logic.py             # Unit tests
```

---

## 🎯 Usage Examples

### Basic Debate

```python
from main import DebateOrchestrator

# Initialize
orchestrator = DebateOrchestrator(project_id="your-project-id")

# Setup agents
orchestrator.setup_agent("Philadelphia")
orchestrator.setup_agent("San Francisco")

# Run debate
orchestrator.run_debate(
    topic="Is Brock Purdy an elite quarterback?",
    agent1_city="Philadelphia",
    agent2_city="San Francisco",
    num_turns=3
)
```

### Custom Agent Response

```python
# Get a single response with tempo and flavor
response = orchestrator.get_response(
    city_name="Philadelphia",
    user_message="What do you think about analytics in football?",
    show_thinking=True
)
print(response)
```

---

## 🔧 Configuration

### City Profiles

All city personalities are defined in `config/city_profiles.json`:

- **Tempo**: Response speed and interruption behavior
- **Evidence Weights**: How each city prioritizes different types of evidence
- **Lexical Style**: City-specific phrases and formality level
- **Sentiment**: Emotional range and volatility

### Available Cities

32 NFL cities are available:
- Philadelphia (aggressive, eye-test focused)
- San Francisco (analytical, data-driven)
- Kansas City (fast, execution-focused)
- New England (strategic, system-focused)
- Buffalo (resilient, long-suffering)
- ...and 27 more!

---

## 🧪 Testing

### Run Core Logic Tests

```bash
python test_core_logic.py
```

This tests:
- AgentFactory (prompt construction)
- TempoEngine (delays and interruptions)
- LexicalInjector (phrase injection)

---

## 🎨 How It Works

### 1. Agent Factory
- Loads city profiles from JSON
- Dynamically constructs system prompts based on evidence weights
- Example: Philadelphia prioritizes "eye test" (0.40) and "toughness" (0.30)

### 2. Tempo Engine
- Calculates realistic response delays with variance
- Determines when to interrupt based on opponent confidence
- Example: Philadelphia (140ms ± 15ms, interrupts at < 0.3 confidence)

### 3. Lexical Injector
- Injects city-specific phrases based on injection rate
- Uses 4 strategies: prefix, suffix, parenthetical, emphasis
- Example: Philadelphia phrases include "grit", "heart", "no excuses"

### 4. Vertex Client
- Uses AnthropicVertex SDK for Claude 3.5 Sonnet
- Sends messages with system prompts and conversation history
- Handles authentication and error management

### 5. Main Orchestrator
- Integrates all components
- Manages multi-turn debates
- Applies tempo delays and lexical flavor to responses

---

## 🐛 Troubleshooting

### "GOOGLE_CLOUD_PROJECT not set"

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
```

### "Permission denied" errors

Make sure you're authenticated:
```bash
gcloud auth application-default login
```

### "City not found" errors

Check available cities:
```python
from src.core.agent_factory import AgentFactory
factory = AgentFactory()
print(factory.get_all_cities())
```

---

## 📝 Example Output

```
============================================================
DEBATE: Is Brock Purdy an elite quarterback?
============================================================

Participants:
  • Philadelphia (Aggressive, Eye Test Focused)
  • San Francisco (Analytical, Data-Driven)

------------------------------------------------------------

🗣️  Philadelphia:
   Elite? Hold on - prove it first. Sure, Purdy's got weapons, but 
   great system doesn't equal elite. Show me the grit when it matters!

--- Turn 1 ---

🗣️  San Francisco:
   The data tells a different story - Purdy's EPA/play ranks in the 
   top 5. When you optimize for system efficiency, you get elite 
   results. That's the framework.

🗣️  Philadelphia:
   [⚡ Philadelphia interrupts!]
   Data means nothing without heart! I'll take the guy who battles 
   through adversity over spreadsheet excellence any day. NO EXCUSES!

...
```

---

## 🚦 Next Steps

1. ✅ Install dependencies (`pip install -r requirements.txt`)
2. ✅ Set up Google Cloud authentication
3. ✅ Run the debate (`python main.py`)
4. 🎯 Experiment with different city combinations
5. 🧪 Test with your own debate topics
6. 🔧 Customize city profiles in `config/city_profiles.json`
