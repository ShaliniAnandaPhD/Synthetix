# TNF Live Test Runbook - December 18, 2025

## Game Info

| Field | Value |
|-------|-------|
| **Matchup** | Los Angeles Rams @ Seattle Seahawks |
| **Date** | Thursday, December 18, 2025 |
| **Kickoff** | 8:15 PM ET / 5:15 PM PT |
| **Broadcast** | Prime Video |
| **Regions** | `los_angeles` (homer), `seattle` (homer) |

---

## Pre-Game Checklist (T-60 min = 7:15 PM ET)

### Infrastructure Check
- [ ] Modal WebSocket server healthy: `modal serve infra/modal_live_ws.py`
- [ ] Verify endpoint responds: `curl https://neuronsystems--neuron-live-ws-live-server.modal.run/health`
- [ ] Check Modal dashboard for errors

### Cache Warming
```bash
# Warm regional caches for both teams
python scripts/warm_game_context.py --game=TNF_DEC18 --home=seattle --away=los_angeles

# Verify cache populated
python -c "from src.core import get_phrase_cache; c = get_phrase_cache(); print(c.get_stats())"
```

### Observability Setup
- [ ] Open W&B dashboard: https://wandb.ai/[your-org]/neuron-live
- [ ] Set up cost alert at $20 max game spend
- [ ] Open Modal logs: https://modal.com/apps/shalini/main

### Creator Notification
- [ ] Notify 3-5 test creators
- [ ] Share WebSocket URL for testing
- [ ] Remind: Save session IDs for post-game analysis

---

## During Game (8:15 PM - ~11:30 PM ET)

### Monitor Dashboard

| Metric | Target | Action if Exceeded |
|--------|--------|-------------------|
| P95 latency | < 500ms | Switch to cache-only |
| Cache hit rate | > 60% | Check cache warming |
| Cost/hour | < $5/hr | Enable throttling |
| Error rate | < 1% | Check logs |

### Key Moments to Watch

| Quarter | Events | Notes |
|---------|--------|-------|
| Q1 | First TD, early turnovers | Calibrate latency baseline |
| Q2 | 2-minute drill | High event density |
| Q3 | Momentum shifts | Team emotion calibration |
| Q4 | Clutch plays, game-deciding | Maximum engagement |

### Real-Time Commands
```bash
# Check current sessions
curl https://neuronsystems--neuron-live-ws-live-server.modal.run/sessions

# View cost status
python -c "from src.core import create_cost_cap; cap = create_cost_cap('TNF_DEC18'); print(cap.get_status())"

# Emergency: Kill all sessions
modal app stop neuron-live-ws
```

---

## Post-Game (Within 24 Hours)

### Data Collection
- [ ] Export all session data from W&B
- [ ] Calculate total game cost
- [ ] Count total events processed
- [ ] Measure average latency by event type

### Analysis
```bash
# Export session metrics
python scripts/export_game_metrics.py --game=TNF_DEC18

# Generate report
python scripts/generate_game_report.py --game=TNF_DEC18
```

### Creator Feedback
- [ ] Send feedback survey to test creators
- [ ] Document any issues they reported
- [ ] Note feature requests

### Retrospective Items
- [ ] What worked well?
- [ ] What needs improvement?
- [ ] Latency issues?
- [ ] Cost surprises?
- [ ] Cache misses?

---

## Emergency Procedures

### 🔴 Latency Spike > 2000ms

1. Check Modal dashboard for cold starts
2. Switch to cache-only mode:
   ```bash
   python -c "from src.reliability import get_feature_flags; f = get_feature_flags(); f.set('live_agent_enabled', False)"
   ```
3. Monitor recovery

### 🔴 Cost Exceeds $20

1. Enable throttling immediately:
   ```bash
   python -c "from src.core import create_cost_cap; cap = create_cost_cap('TNF_DEC18'); cap.enable_throttle()"
   ```
2. Switch to text-only (no TTS)
3. Notify test creators

### 🔴 WebSocket Disconnects

1. Sessions auto-reconnect (client-side)
2. Tell creators to refresh if stuck
3. Check Modal for errors:
   ```bash
   modal app logs neuron-live-ws
   ```

### 🔴 Complete Failure

1. Stop Modal app: `modal app stop neuron-live-ws`
2. Notify creators: "Technical difficulties, will retry next game"
3. Capture all logs for debugging

---

## Contacts

| Role | Name | Contact |
|------|------|---------|
| Lead Engineer | [You] | [Your contact] |
| Test Creators | [List] | [Discord/Slack] |
| Modal Support | - | https://modal.com/docs |

---

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Uptime | 95% | 99% |
| P95 Latency | < 1000ms | < 500ms |
| Cache Hit Rate | > 50% | > 70% |
| Total Cost | < $25 | < $15 |
| Creator Satisfaction | 3/5 | 4/5 |
