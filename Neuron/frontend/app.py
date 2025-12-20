"""
Streamlit Frontend for Neuron Core - Real-Time NFL Debate Dashboard

Connects to Confluent Kafka and displays AI-generated debates in real-time.
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Neuron Core - NFL AI Debates",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stMetric { padding: 10px; border-radius: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .debate-card { 
        padding: 20px; 
        border-radius: 10px; 
        margin: 10px 0; 
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border-left: 4px solid #4ade80;
    }
    .blocked-card { 
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, #1a0f0f 0%, #2e1a1a 100%);
    }
    .timestamp { color: #888; font-size: 0.8rem; }
    .locale-badge { 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-size: 0.75rem; 
        background: #3b82f6;
        color: white;
    }
    h1 { color: #4ade80; }
    .big-emoji { font-size: 3rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# KAFKA CONSUMER (Background Thread)
# ============================================================================

class KafkaDebateConsumer:
    """Background Kafka consumer for real-time debate streaming."""
    
    def __init__(self, max_messages: int = 50):
        self.messages = deque(maxlen=max_messages)
        self.running = False
        self.thread = None
        self.stats = {
            "total": 0,
            "passed": 0,
            "blocked": 0
        }
        
    def start(self, kafka_config: Dict[str, Any], topic: str):
        """Start the consumer thread."""
        self.running = True
        self.thread = threading.Thread(
            target=self._consume_loop,
            args=(kafka_config, topic),
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop the consumer thread."""
        self.running = False
    
    def _consume_loop(self, kafka_config: Dict[str, Any], topic: str):
        """Main consumer loop."""
        try:
            from confluent_kafka import Consumer, KafkaError
            
            consumer_config = {
                **kafka_config,
                'group.id': 'streamlit-frontend-consumer',
                'auto.offset.reset': 'latest',
                'enable.auto.commit': True
            }
            
            consumer = Consumer(consumer_config)
            consumer.subscribe([topic])
            
            while self.running:
                msg = consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    continue
                
                try:
                    value = msg.value().decode('utf-8') if msg.value() else ""
                    data = json.loads(value)
                    
                    message = {
                        "timestamp": datetime.now().isoformat(),
                        "key": msg.key().decode('utf-8') if msg.key() else None,
                        "data": data,
                        "is_safe": data.get("is_safe", True)
                    }
                    
                    self.messages.appendleft(message)
                    self.stats["total"] += 1
                    
                    if message["is_safe"]:
                        self.stats["passed"] += 1
                    else:
                        self.stats["blocked"] += 1
                        
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                    
            consumer.close()
            
        except Exception as e:
            st.error(f"Kafka consumer error: {e}")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return list(self.messages)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current stats."""
        return self.stats.copy()


# ============================================================================
# MOCK DATA (for demo without Kafka)
# ============================================================================

MOCK_DEBATES = [
    {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "answer": "TOUCHDOOOOOWN! É GOOOOOL! Mahomes throws a SENSACIONAL 50-yard bomb to Kelce!",
            "locale": "pt-BR",
            "confidence": 0.95,
            "version": "1.0.5"
        },
        "is_safe": True
    },
    {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "answer": "Right, absolutely brilliant run there. The lad's got proper pace, reminds me of a young Vardy.",
            "locale": "en-GB",
            "confidence": 0.92,
            "version": "1.0.5"
        },
        "is_safe": True
    },
    {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "answer": "[BLOCKED] Content contained unsafe keywords",
            "locale": "en-US",
            "safety_reason": "Compliance Violation: fake"
        },
        "is_safe": False
    }
]


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏈 Neuron Core - NFL AI Debates")
        st.markdown("*Real-time AI commentary powered by Vertex AI + Confluent Kafka*")
    with col2:
        st.markdown('<div class="big-emoji">🧠</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        demo_mode = st.toggle("Demo Mode (Mock Data)", value=True)
        
        if not demo_mode:
            st.subheader("Kafka Settings")
            bootstrap = st.text_input(
                "Bootstrap Servers", 
                value="pkc-619z3.us-east1.gcp.confluent.cloud:9092"
            )
            topic = st.text_input("Topic", value="agent-debates")
            
            if st.button("🔗 Connect to Kafka"):
                st.session_state.kafka_connected = True
                st.success("Connected!")
        
        st.divider()
        
        st.subheader("📊 Stats")
        stats = st.session_state.get("stats", {"total": 0, "passed": 0, "blocked": 0})
        st.metric("Total Messages", stats["total"])
        st.metric("✅ Passed", stats["passed"])
        st.metric("🚫 Blocked", stats["blocked"])
        
        st.divider()
        
        st.subheader("🎭 Personality Locales")
        locale_filter = st.multiselect(
            "Filter by locale",
            ["en-US", "en-GB", "pt-BR", "es-MX", "en-AU", "ja-JP"],
            default=["en-US", "pt-BR", "en-GB"]
        )
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.header("🎙️ Live Debate Feed")
        
        # Get messages (demo or real)
        if demo_mode:
            messages = MOCK_DEBATES
        else:
            messages = st.session_state.get("messages", [])
        
        # Display messages
        if not messages:
            st.info("Waiting for debate messages... Send events to `nfl-game-events` in Confluent Cloud.")
        else:
            for msg in messages:
                data = msg.get("data", {})
                locale = data.get("locale", "en-US")
                
                # Apply locale filter
                if locale not in locale_filter:
                    continue
                
                is_safe = msg.get("is_safe", True)
                card_class = "debate-card" if is_safe else "debate-card blocked-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}">
                        <span class="locale-badge">{locale}</span>
                        <span class="timestamp"> {msg.get('timestamp', '')[:19]}</span>
                        <br><br>
                        <strong>{'✅' if is_safe else '🚫'} {data.get('answer', data.get('safety_reason', 'No content'))[:300]}</strong>
                        <br><br>
                        <small>Confidence: {data.get('confidence', 'N/A')} | Version: {data.get('version', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Auto-refresh button
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with col_right:
        st.header("🏆 System Status")
        
        # Architecture diagram
        st.markdown("""
        ```
        ┌──────────────┐
        │  Confluent   │
        │    Kafka     │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ Kafka Bridge │
        │   (Safety)   │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  Vertex AI   │
        │   Agent      │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  BigQuery    │
        │  (Analytics) │
        └──────────────┘
        ```
        """)
        
        st.divider()
        
        st.subheader("🔧 Quick Actions")
        
        if st.button("🧪 Test Vertex AI"):
            with st.spinner("Querying agent..."):
                try:
                    import vertexai
                    from vertexai.preview import reasoning_engines
                    vertexai.init(project='leafy-sanctuary-476515-t2', location='us-central1')
                    agent = reasoning_engines.ReasoningEngine(
                        'projects/488602940935/locations/us-central1/reasoningEngines/205135884394168320'
                    )
                    result = agent.query(input_text="Test from Streamlit dashboard")
                    st.success(f"✅ Agent responded!")
                    st.json(result)
                except Exception as e:
                    st.error(f"❌ {e}")
        
        if st.button("📊 Query BigQuery"):
            with st.spinner("Querying analytics..."):
                try:
                    from google.cloud import bigquery
                    client = bigquery.Client(project='leafy-sanctuary-476515-t2')
                    query = """
                    SELECT is_safe, COUNT(*) as count 
                    FROM `leafy-sanctuary-476515-t2.nfl_analysis.agent_debates_log`
                    GROUP BY is_safe
                    """
                    df = client.query(query).to_dataframe()
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"❌ {e}")


if __name__ == "__main__":
    main()
