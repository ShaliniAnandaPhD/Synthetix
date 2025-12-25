# NEURON Mission Control 🧠

## Confluent x Google Cloud Hackathon Submission

**Live Demo:** https://neuron-mission-control.web.app/dashboard.html

---

## 🎬 10-Second Proof: Data in Motion

![Demo Video](data_in_motion_demo_1766697843667.webp)

**What the video shows:**
1. Click "TD CHIEFS" → Event flows through Kafka → Multi-agent debate appears in <300ms
2. Click "INT RAVENS" → Same real-time pipeline → New debate cards stream in

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    NEURON MISSION CONTROL                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CONFLUENT     │───▶│  GOOGLE CLOUD   │───▶│    FIREBASE     │
│     KAFKA       │    │    CLOUD RUN    │    │     HOSTING     │
│                 │    │                 │    │                 │
│ • Event Stream  │    │ • Swarm AI      │    │ • React UI      │
│ • Real-time     │    │ • Cloud TTS     │    │ • SSE Streaming │
│   Ingestion     │    │ • Multi-Agent   │    │ • Global CDN    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                    Real-time event pipeline
```

---

## 📊 Quantified Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **API Latency** | 260ms | End-to-end (request → dual-agent response) |
| **Agent Count** | 2 simultaneous | Homer (fanatic) + Skeptic (analyst) |
| **Voice Languages** | 7 | US, BR, MX, AU, GB, DE, JP |
| **Cold Start** | <3s | Cloud Run scales from 0 |

---

## 🏃 How to Run

### Prerequisites
- Google Cloud account with billing enabled
- Firebase CLI installed
- Python 3.9+

### Local Development
```bash
# Clone the repo
git clone https://github.com/ShaliniAnandaPhD/Synthetix.git
cd Synthetix/Neuron

# Start the API
python3 services/dashboard_api.py

# Serve frontend (in another terminal)
cd frontend && python3 -m http.server 3000

# Open http://localhost:3000/dashboard.html
```

### Deploy to Production
```bash
# Deploy backend to Cloud Run
gcloud run deploy neuron-api --source . --region us-central1 --allow-unauthenticated

# Deploy frontend to Firebase
firebase deploy --only hosting
```

---

## 🚀 What's Next

1. **Live NFL Integration** — Connect to ESPN API for real game events instead of simulated buttons
2. **Kafka Producer** — Deploy Confluent Cloud producer to ingest events from external sources

---

## 🏆 Key Capabilities

| Feature | Description |
|---------|-------------|
| 🧠 Object Permanence | Context persists across server restarts via Firestore |
| 🎬 Referee Bot | Multimodal video analysis + NFL rulebook citations |
| 🦎 Cultural Routing | Dynamic personality swap based on user locale |
| 🐝 Swarm Intelligence | Multi-agent debate (Homer vs Skeptic) in real-time |
| 🛡️ Circuit Breakers | ValidatorAgent blocks toxic/hallucinated content |

---

**Built with:** Confluent Kafka • Google Cloud Run • Google Cloud TTS • Firebase Hosting • React
