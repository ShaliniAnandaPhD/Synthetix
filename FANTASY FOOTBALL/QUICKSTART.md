# Quick Reference - AI Fantasy Football Debate System

## 🚀 Run the System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'

# 3. Authenticate
gcloud auth application-default login

# 4. Run debate
python main.py
```

---

## 📦 Project Files

| File | Purpose |
|------|---------|
| `src/core/agent_factory.py` | Dynamic system prompt construction |
| `src/core/tempo_engine.py` | Response timing and interruptions |
| `src/core/lexical_injector.py` | City-specific phrase injection |
| `src/llm/vertex_client.py` | Claude 3.5 Sonnet on Vertex AI |
| `main.py` | Debate orchestrator |
| `config/city_profiles.json` | 32 city configurations |

---

## 🏙️ Available Cities

**Aggressive (Fast, High Interruption):**
- Philadelphia (Eye test focused, 0.9 aggression)
- Baltimore (Physical, defense-first, 0.82 aggression)
- New England (Strategic, system-focused, 0.85 aggression)

**Analytical (Deliberate, Data-Driven):**
- San Francisco (Advanced stats focused, 0.5 aggression)
- Los Angeles Chargers (Efficiency metrics, 0.52 aggression)

**All 32 NFL Cities Available** - Each with unique personality!

---

## 🔑 Key Components

### AgentFactory
```python
from src.core.agent_factory import AgentFactory

factory = AgentFactory()
prompt = factory.construct_system_prompt("Philadelphia")
# Returns system prompt with evidence weights prioritized
```

### TempoEngine
```python
from src.core.tempo_engine import TempoEngine

tempo = TempoEngine()
delay = tempo.get_delay("Philadelphia")  # ~0.14s
should_interrupt = tempo.check_interruption("Philadelphia", 0.3)  # True
```

### LexicalInjector
```python
from src.core import lexical_injector

text = "That was a great play."
flavored = lexical_injector.inject_flavor(text, "Philadelphia")
# Possible: "That was a great play - that's heart right there."
```

### VertexAgent
```python
from src.llm.vertex_client import VertexAgent

agent = VertexAgent(project_id="my-project")
response = agent.send_message(
    system_instruction="You are...",
    user_message="Is Brock Purdy elite?"
)
```

---

## 🎯 Custom Debates

```python
from main import DebateOrchestrator

# Initialize
orchestrator = DebateOrchestrator(project_id="my-project")

# Setup any two cities
orchestrator.setup_agent("Buffalo")
orchestrator.setup_agent("Kansas City")

# Run custom debate
orchestrator.run_debate(
    topic="Who deserves MVP?",
    agent1_city="Buffalo",
    agent2_city="Kansas City",
    num_turns=5
)
```

---

## 🧪 Testing

```bash
# Test core logic (no API calls)
python test_core_logic.py

# Expected output:
# AgentFactory: ✓ PASS
# TempoEngine: ✓ PASS
# LexicalInjector: ✓ PASS
```

---

## 🐛 Common Issues

**"Module not found"**
```bash
# Ensure you're in project root
cd "FANTASY FOOTBALL"
python main.py
```

**"GOOGLE_CLOUD_PROJECT not set"**
```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
```

**"Permission denied"**
```bash
gcloud auth application-default login
```

---

## 📊 City Comparison

| City | Base Delay | Aggression | Top Priority | Formality |
|------|------------|------------|--------------|-----------|
| Philadelphia | 140ms | 0.90 | Eye Test (0.40) | 0.45 |
| San Francisco | 180ms | 0.50 | Advanced Stats (0.40) | 0.70 |
| Kansas City | 120ms | 0.70 | Advanced Stats (0.30) | 0.55 |
| New England | 160ms | 0.85 | Advanced Stats (0.35) | 0.72 |
| Buffalo | 220ms | 0.60 | Eye Test (0.28) | 0.40 |

---

## 📝 Example Output

```
============================================================
AI FANTASY FOOTBALL DEBATE SYSTEM
============================================================

📋 Setting up agents...
✓ Setup Philadelphia agent
✓ Setup San Francisco agent

============================================================
DEBATE: Is Brock Purdy an elite quarterback?
============================================================

Participants:
  • Philadelphia (Aggressive, Eye Test Focused)
  • San Francisco (Analytical, Data-Driven)

------------------------------------------------------------

🗣️  Philadelphia:
   You know, prove it - Elite? Sure, Purdy's got weapons, but 
   a great system doesn't equal elite QB play!

--- Turn 1 ---

🗣️  San Francisco:
   The data framework suggests otherwise - Purdy's EPA/play 
   efficiency ranks in the top 5...
```

---

## 🎓 For More Details

- **Full Setup**: See [SETUP.md](file:///Users/shaliniananda/FANTASY%20FOOTBALL/SETUP.md)
- **Implementation Details**: See walkthrough.md artifact
- **City Configs**: See [config/city_profiles.json](file:///Users/shaliniananda/FANTASY%20FOOTBALL/config/city_profiles.json)
